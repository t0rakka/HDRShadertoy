/*
    MANGO Multimedia Development Platform
    Copyright (C) 2012-2026 Twilight Finland 3D Oy Ltd. All rights reserved.

    Vulkan Shadertoy compute shader player — Float16 RenderTarget, HDR resolve.
    Supports single-pass .comp files and multipass effects (#SHADER / #CHANNEL).
*/
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#define MANGO_IMPLEMENT_MAIN

#include <mango/core/core.hpp>
#include <mango/filesystem/filesystem.hpp>
#include <mango/math/math.hpp>
#include <mango/vulkan/vulkan.hpp>

using namespace mango;
using namespace mango::math;
using namespace mango::filesystem;
using namespace mango::vulkan;

namespace
{

    constexpr int kChannelCount = 4;

    enum class PassId : int
    {
        A = 0,
        B = 1,
        C = 2,
        D = 3,
        Image = 4,
        Count = 5,
        Invalid = -1,
    };

    const char* passIdName(PassId id)
    {
        switch (id)
        {
            case PassId::A: return "A";
            case PassId::B: return "B";
            case PassId::C: return "C";
            case PassId::D: return "D";
            case PassId::Image: return "Image";
            default: return "?";
        }
    }

    PassId parsePassId(std::string_view s)
    {
        if (s == "A" || s == "a") return PassId::A;
        if (s == "B" || s == "b") return PassId::B;
        if (s == "C" || s == "c") return PassId::C;
        if (s == "D" || s == "d") return PassId::D;
        if (s == "Image" || s == "IMAGE" || s == "image") return PassId::Image;
        return PassId::Invalid;
    }

    int passSortKey(PassId id)
    {
        return int(id);
    }

    struct ChannelBinding
    {
        PassId pass = PassId::Invalid;
        int slot = 0;
        PassId source = PassId::Invalid;
    };

    struct EffectPass
    {
        PassId id = PassId::Image;
        size_t spirvOffset = 0;
        size_t spirvCount = 0;

        std::span<const u32> view(const std::vector<u32>& spirv) const
        {
            return { spirv.data() + spirvOffset, spirvCount };
        }
    };

    struct Effect
    {
        std::string name;
        std::vector<EffectPass> passes;
        std::vector<ChannelBinding> channels;

        bool hasPass(PassId id) const
        {
            for (const EffectPass& pass : passes)
            {
                if (pass.id == id)
                {
                    return true;
                }
            }
            return false;
        }

        const EffectPass* findPass(PassId id) const
        {
            for (const EffectPass& pass : passes)
            {
                if (pass.id == id)
                {
                    return &pass;
                }
            }
            return nullptr;
        }
    };

    struct ShaderLibrary
    {
        std::vector<u32> spirv;
        std::vector<Effect> effects;
    };

    // Matches layout(push_constant) in every .comp shader.
    struct ShadertoyPush
    {
        float time = 0.0f;
        float pad0 = 0.0f;
        float32x2 resolution {};
        float32x4 mouse {};
    };

    static_assert(sizeof(ShadertoyPush) <= 128, "push constants must fit default limit");

    std::string trim(std::string_view s)
    {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n'))
        {
            s.remove_prefix(1);
        }
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n'))
        {
            s.remove_suffix(1);
        }
        return std::string(s);
    }

    std::string stripChannelDirectives(const std::string& source)
    {
        std::string out;
        out.reserve(source.size());
        size_t i = 0;
        while (i < source.size())
        {
            size_t lineEnd = source.find('\n', i);
            if (lineEnd == std::string::npos)
            {
                lineEnd = source.size();
            }

            std::string_view line(source.data() + i, lineEnd - i);
            const std::string trimmed = trim(line);
            if (trimmed.rfind("#CHANNEL", 0) != 0)
            {
                out.append(source, i, lineEnd - i);
                if (lineEnd < source.size())
                {
                    out.push_back('\n');
                }
            }

            i = (lineEnd < source.size()) ? lineEnd + 1 : lineEnd;
        }
        return out;
    }

    std::vector<ChannelBinding> parseChannelDirectives(const std::string& source)
    {
        std::vector<ChannelBinding> channels;
        size_t i = 0;
        while (i < source.size())
        {
            size_t lineEnd = source.find('\n', i);
            if (lineEnd == std::string::npos)
            {
                lineEnd = source.size();
            }

            std::string_view line(source.data() + i, lineEnd - i);
            const std::string trimmed = trim(line);
            if (trimmed.rfind("#CHANNEL", 0) == 0)
            {
                const size_t open = trimmed.find('(');
                const size_t close = trimmed.rfind(')');
                if (open != std::string::npos && close != std::string::npos && close > open)
                {
                    const std::string args = trim(trimmed.substr(open + 1, close - open - 1));
                    // pass, slot, source
                    size_t c1 = args.find(',');
                    size_t c2 = (c1 == std::string::npos) ? std::string::npos : args.find(',', c1 + 1);
                    if (c1 != std::string::npos && c2 != std::string::npos)
                    {
                        const PassId pass = parsePassId(trim(args.substr(0, c1)));
                        const int slot = std::atoi(trim(args.substr(c1 + 1, c2 - c1 - 1)).c_str());
                        const PassId sourcePass = parsePassId(trim(args.substr(c2 + 1)));
                        if (pass != PassId::Invalid && sourcePass != PassId::Invalid && slot >= 0 && slot < kChannelCount)
                        {
                            channels.push_back({ pass, slot, sourcePass });
                        }
                        else
                        {
                            printLine(Print::Error, "Invalid #CHANNEL directive: {}", trimmed);
                        }
                    }
                }
            }

            i = (lineEnd < source.size()) ? lineEnd + 1 : lineEnd;
        }
        return channels;
    }

    struct ParsedSection
    {
        PassId id = PassId::Image;
        std::string source;
    };

    std::vector<ParsedSection> parseShaderSections(const std::string& source)
    {
        std::vector<ParsedSection> sections;
        const std::string marker = "#SHADER";
        size_t pos = 0;
        bool found = false;

        while (pos < source.size())
        {
            const size_t markerPos = source.find(marker, pos);
            if (markerPos == std::string::npos)
            {
                break;
            }

            found = true;
            const size_t open = source.find('(', markerPos);
            const size_t close = source.find(')', open == std::string::npos ? markerPos : open);
            if (open == std::string::npos || close == std::string::npos)
            {
                printLine(Print::Error, "Malformed #SHADER directive");
                break;
            }

            const PassId id = parsePassId(trim(source.substr(open + 1, close - open - 1)));
            if (id == PassId::Invalid)
            {
                printLine(Print::Error, "Unknown #SHADER id");
                pos = close + 1;
                continue;
            }

            size_t contentStart = close + 1;
            while (contentStart < source.size() && (source[contentStart] == '\r' || source[contentStart] == '\n'))
            {
                ++contentStart;
            }

            const size_t nextMarker = source.find(marker, contentStart);
            const size_t contentEnd = (nextMarker == std::string::npos) ? source.size() : nextMarker;
            sections.push_back({ id, stripChannelDirectives(source.substr(contentStart, contentEnd - contentStart)) });
            pos = contentEnd;
        }

        if (!found)
        {
            sections.push_back({ PassId::Image, stripChannelDirectives(source) });
        }

        std::sort(sections.begin(), sections.end(), [](const ParsedSection& a, const ParsedSection& b)
        {
            return passSortKey(a.id) < passSortKey(b.id);
        });

        return sections;
    }

    ShaderLibrary loadCompiledShaders(Path& path)
    {
        Path shadersDir(path, "shaders/");
        const FileIndex& index = shadersDir.getIndex();

        std::vector<std::string> names;
        for (const FileInfo& info : index)
        {
            if (info.isDirectory())
            {
                continue;
            }

            if (getExtension(info.name) == ".comp")
            {
                names.push_back(info.name);
            }
        }

        std::sort(names.begin(), names.end());

        if (names.empty())
        {
            printLine(Print::Error, "No .comp files in {}", shadersDir.pathname());
            return {};
        }

        Compiler compiler;
        ShaderLibrary library;

        const int total = int(names.size());
        for (int i = 0; i < total; ++i)
        {
            const std::string& name = names[i];
            printLine("[Compiling {:02}/{:02}] {}", i + 1, total, name);

            File file(shadersDir, name);
            if (!file.size())
            {
                printLine(Print::Error, "Failed to read shader: {}/{}", shadersDir.pathname(), name);
                continue;
            }

            const std::string source(reinterpret_cast<const char*>(file.data()), size_t(file.size()));
            const std::vector<ChannelBinding> channels = parseChannelDirectives(source);
            const std::vector<ParsedSection> sections = parseShaderSections(source);
            if (sections.empty())
            {
                printLine(Print::Error, "No shader sections in {}", name);
                continue;
            }

            Effect effect;
            effect.name = name;
            effect.channels = channels;

            bool ok = true;
            for (const ParsedSection& section : sections)
            {
                Shader shader = compiler.compile(section.source, ShaderStage::Compute);
                if (!shader)
                {
                    printLine(Print::Error, "Compute compile failed: {} [{}]", name, passIdName(section.id));
                    if (!shader.log.empty())
                    {
                        printLine(Print::Error, "{}", shader.log);
                    }
                    ok = false;
                    break;
                }

                const size_t offset = library.spirv.size();
                library.spirv.insert(library.spirv.end(), shader.spirv.begin(), shader.spirv.end());
                effect.passes.push_back({
                    .id = section.id,
                    .spirvOffset = offset,
                    .spirvCount = shader.spirv.size(),
                });
            }

            if (ok && !effect.passes.empty())
            {
                if (effect.passes.size() > 1)
                {
                    printLine(Print::Info, "  multipass: {} sections", effect.passes.size());
                }
                library.effects.push_back(std::move(effect));
            }
        }

        return library;
    }

    size_t findEffectIndex(const ShaderLibrary& library, const std::string& name)
    {
        for (size_t i = 0; i < library.effects.size(); ++i)
        {
            if (library.effects[i].name == name)
            {
                return i;
            }
        }
        return size_t(-1);
    }

    std::string effectKeyFromArg(std::string arg)
    {
        // Accept AlienBooger, AlienBooger.comp, data/shaders/AlienBooger.comp, etc.
        while (!arg.empty() && (arg.back() == '/' || arg.back() == '\\'))
        {
            arg.pop_back();
        }
        const size_t slash = arg.find_last_of("/\\");
        if (slash != std::string::npos)
        {
            arg = arg.substr(slash + 1);
        }
        if (getExtension(arg).empty())
        {
            arg += ".comp";
        }
        return arg;
    }

    VkImageMemoryBarrier makeImageBarrier(VkImage image,
        VkAccessFlags srcAccess, VkAccessFlags dstAccess,
        VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        return VkImageMemoryBarrier
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = srcAccess,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };
    }

} // namespace

class ShadertoyWindow : public VulkanWindow
{
protected:
    struct PassGpu
    {
        PassId id = PassId::Image;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    std::unique_ptr<Allocator> m_allocator;
    std::unique_ptr<RenderTarget> m_renderTarget;
    std::array<std::unique_ptr<RenderTarget>, 4> m_buffers {}; // A..D
    std::unique_ptr<RenderTarget> m_dummyChannel;

    OutputTransformOptions m_outputOptions {};
    bool m_hdrSwapchain = false;

    ShaderLibrary m_library;
    size_t m_effectIndex = 0;

    VkDescriptorSetLayout m_descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;

    std::vector<PassGpu> m_passes;

    float m_frameTimeMs = 0.0f;
    std::string m_statusLine;

    float32x4 m_mouse {};
    bool m_mouseDown = false;

public:
    ShadertoyWindow(VkInstance instance, int width, int height, const VulkanDeviceConfig* config,
                     ShaderLibrary library, size_t initialIndex = 0)
        : VulkanWindow(instance, width, height, 0, config)
        , m_library(std::move(library))
        , m_effectIndex(std::min(initialIndex, m_library.effects.empty() ? 0 : m_library.effects.size() - 1))
    {
    }

    void onDeviceReady() override
    {
        m_allocator = std::make_unique<Allocator>(instance(), m_physicalDevice, m_device, VK_API_VERSION_1_3);

        m_hdrSwapchain = isHDR(surfaceFormat());
        m_outputOptions = defaultOutputOptions(surfaceFormat(), 1.0f);

        printLine("Shader player: swapchain {} / {} (HDR: {})",
            getString(surfaceFormat().format),
            getString(surfaceFormat().colorSpace),
            m_hdrSwapchain ? "yes" : "no");

        printLine("Use LEFT/RIGHT to cycle shaders.");

        createComputeResources();
        recreateRenderTargets(swapchainExtent());
        applyEffect(m_effectIndex);
        updateStatusLine();
    }

    void onSwapchainResize(VkExtent2D extent) override
    {
        m_hdrSwapchain = isHDR(surfaceFormat());
        m_outputOptions = defaultOutputOptions(surfaceFormat(), m_outputOptions.exposure);
        recreateRenderTargets(extent);
        updatePassDescriptors();
        updateStatusLine();
    }

    ~ShadertoyWindow()
    {
        if (m_device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device);
            destroyPassGpu();
            destroyComputeResources();
            m_renderTarget.reset();
            for (auto& buffer : m_buffers)
            {
                buffer.reset();
            }
            m_dummyChannel.reset();
            m_allocator.reset();
        }
    }

    void createComputeResources()
    {
        if (m_library.effects.empty())
        {
            printLine(Print::Error, "No compiled effects available.");
            return;
        }

        std::array<VkDescriptorSetLayoutBinding, 1 + kChannelCount> bindings {};
        bindings[0] = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        for (int i = 0; i < kChannelCount; ++i)
        {
            bindings[1 + i] = {
                .binding = u32(1 + i),
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        };

        vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorLayout);

        VkPushConstantRange pushRange =
        {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(ShadertoyPush),
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_descriptorLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange,
        };

        vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);

        // Enough for one effect with up to 5 passes (A-D + Image).
        constexpr u32 kMaxPasses = 5;
        std::array<VkDescriptorPoolSize, 2> poolSizes = {{
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxPasses },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxPasses * kChannelCount },
        }};

        VkDescriptorPoolCreateInfo poolInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = kMaxPasses,
            .poolSizeCount = u32(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);

        VkSamplerCreateInfo samplerInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod = 0.0f,
        };
        vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler);
    }

    void destroyPassGpu()
    {
        for (PassGpu& pass : m_passes)
        {
            if (pass.pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, pass.pipeline, nullptr);
                pass.pipeline = VK_NULL_HANDLE;
            }
            if (pass.module != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(m_device, pass.module, nullptr);
                pass.module = VK_NULL_HANDLE;
            }
            pass.set = VK_NULL_HANDLE;
        }
        m_passes.clear();

        if (m_descriptorPool != VK_NULL_HANDLE)
        {
            vkResetDescriptorPool(m_device, m_descriptorPool, 0);
        }
    }

    void applyEffect(size_t index)
    {
        if (m_library.effects.empty() || index >= m_library.effects.size() || !m_pipelineLayout)
        {
            return;
        }

        vkDeviceWaitIdle(m_device);
        destroyPassGpu();

        m_effectIndex = index;
        const Effect& effect = m_library.effects[m_effectIndex];

        ensureBufferTargets(swapchainExtent());

        for (const EffectPass& compiled : effect.passes)
        {
            PassGpu gpu;
            gpu.id = compiled.id;

            const std::span<const u32> spirvSpan = compiled.view(m_library.spirv);
            if (spirvSpan.empty())
            {
                printLine(Print::Error, "Empty SPIR-V: {} [{}]", effect.name, passIdName(compiled.id));
                continue;
            }

            const std::vector<u32> spirv(spirvSpan.begin(), spirvSpan.end());
            gpu.module = Compiler::createShaderModule(m_device, spirv);

            VkPipelineShaderStageCreateInfo stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = gpu.module,
                .pName = "main",
            };

            VkComputePipelineCreateInfo pipelineInfo =
            {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = stage,
                .layout = m_pipelineLayout,
            };

            vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &gpu.pipeline);
            if (gpu.pipeline == VK_NULL_HANDLE)
            {
                printLine(Print::Error, "vkCreateComputePipelines failed: {} [{}]", effect.name, passIdName(compiled.id));
                if (gpu.module != VK_NULL_HANDLE)
                {
                    vkDestroyShaderModule(m_device, gpu.module, nullptr);
                }
                continue;
            }

            VkDescriptorSetAllocateInfo allocInfo =
            {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = m_descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &m_descriptorLayout,
            };

            VkResult result = vkAllocateDescriptorSets(m_device, &allocInfo, &gpu.set);
            if (result != VK_SUCCESS)
            {
                printLine(Print::Error, "vkAllocateDescriptorSets: {}", getString(result));
                vkDestroyPipeline(m_device, gpu.pipeline, nullptr);
                vkDestroyShaderModule(m_device, gpu.module, nullptr);
                continue;
            }

            m_passes.push_back(gpu);
        }

        updatePassDescriptors();

        printLine(Print::Info, "Effect [{}/{}]: {} ({} pass{})",
            m_effectIndex + 1, m_library.effects.size(), effect.name,
            m_passes.size(), m_passes.size() == 1 ? "" : "es");
        updateStatusLine();
    }

    void cycleShader(int delta)
    {
        if (m_library.effects.size() < 2)
        {
            return;
        }

        const size_t count = m_library.effects.size();
        const size_t next = (m_effectIndex + count + size_t(delta)) % count;
        applyEffect(next);
    }

    void destroyComputeResources()
    {
        destroyPassGpu();

        if (m_sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(m_device, m_sampler, nullptr);
            m_sampler = VK_NULL_HANDLE;
        }

        if (m_descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }

        if (m_pipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }

        if (m_descriptorLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);
            m_descriptorLayout = VK_NULL_HANDLE;
        }
    }

    RenderTarget* bufferTarget(PassId id) const
    {
        const int index = int(id);
        if (index < 0 || index >= 4)
        {
            return nullptr;
        }
        return m_buffers[size_t(index)].get();
    }

    RenderTarget* passTarget(PassId id) const
    {
        if (id == PassId::Image)
        {
            return m_renderTarget.get();
        }
        return bufferTarget(id);
    }

    VkImageView channelView(PassId source) const
    {
        RenderTarget* rt = passTarget(source);
        if (rt && *rt)
        {
            return rt->view();
        }
        return m_dummyChannel && *m_dummyChannel ? m_dummyChannel->view() : VK_NULL_HANDLE;
    }

    PassId channelSource(const Effect& effect, PassId pass, int slot) const
    {
        for (const ChannelBinding& binding : effect.channels)
        {
            if (binding.pass == pass && binding.slot == slot)
            {
                return binding.source;
            }
        }
        return PassId::Invalid;
    }

    void updatePassDescriptors()
    {
        if (!m_renderTarget || !(*m_renderTarget) || !m_dummyChannel || !(*m_dummyChannel) || m_passes.empty())
        {
            return;
        }

        if (m_library.effects.empty())
        {
            return;
        }

        const Effect& effect = m_library.effects[m_effectIndex];

        for (PassGpu& pass : m_passes)
        {
            RenderTarget* target = passTarget(pass.id);
            if (!target || !(*target) || pass.set == VK_NULL_HANDLE)
            {
                continue;
            }

            VkDescriptorImageInfo storageInfo =
            {
                .imageView = target->view(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            std::array<VkDescriptorImageInfo, kChannelCount> channelInfos {};
            std::array<VkWriteDescriptorSet, 1 + kChannelCount> writes {};

            writes[0] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pass.set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageInfo,
            };

            for (int slot = 0; slot < kChannelCount; ++slot)
            {
                const PassId source = channelSource(effect, pass.id, slot);
                VkImageView view = m_dummyChannel->view();
                if (source != PassId::Invalid)
                {
                    const VkImageView srcView = channelView(source);
                    if (srcView != VK_NULL_HANDLE)
                    {
                        view = srcView;
                    }
                }

                channelInfos[size_t(slot)] = {
                    .sampler = m_sampler,
                    .imageView = view,
                    // GENERAL is valid for sampling and matches RenderTarget compute path.
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                };

                writes[1 + slot] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = pass.set,
                    .dstBinding = u32(1 + slot),
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &channelInfos[size_t(slot)],
                };
            }

            vkUpdateDescriptorSets(m_device, u32(writes.size()), writes.data(), 0, nullptr);
        }
    }

    std::unique_ptr<RenderTarget> makeFloat16Target(VkExtent2D extent)
    {
        return std::make_unique<RenderTarget>(RenderTarget::CreateInfo
        {
            .device = m_device,
            .allocator = m_allocator.get(),
            .queue = m_graphicsQueue,
            .queueFamily = m_graphicsQueueFamilyIndex,
            .format = RenderTargetFormat::Float16,
            .extent = extent,
        });
    }

    void ensureBufferTargets(VkExtent2D extent)
    {
        if (!m_allocator || extent.width == 0 || extent.height == 0 || m_library.effects.empty())
        {
            return;
        }

        const Effect& effect = m_library.effects[m_effectIndex];
        for (int i = 0; i < 4; ++i)
        {
            const PassId id = PassId(i);
            if (effect.hasPass(id))
            {
                if (!m_buffers[size_t(i)] || !(*m_buffers[size_t(i)]) ||
                    m_buffers[size_t(i)]->extent().width != extent.width ||
                    m_buffers[size_t(i)]->extent().height != extent.height)
                {
                    m_buffers[size_t(i)] = makeFloat16Target(extent);
                }
            }
        }
    }

    void recreateRenderTargets(VkExtent2D extent)
    {
        if (!m_allocator || extent.width == 0 || extent.height == 0)
        {
            return;
        }

        m_renderTarget = makeFloat16Target(extent);
        m_dummyChannel = makeFloat16Target(VkExtent2D{ 1, 1 });
        ensureBufferTargets(extent);
    }

    void updateStatusLine()
    {
        const VkExtent2D render = swapchainExtent();
        const char* hdrLabel = m_hdrSwapchain ? "HDR" : "SDR (tonemap fallback)";
        const std::string name = m_library.effects.empty() ? "none" : m_library.effects[m_effectIndex].name;

        m_statusLine = fmt::format("{}  {}x{}  {}  Float16 RT  [{}/{}]",
            name, render.width, render.height, hdrLabel,
            m_effectIndex + 1, m_library.effects.size());
    }

    float32x2 clientToRender(float x, float y) const
    {
        const VkExtent2D render = swapchainExtent();
        const math::int32x2 client = getClientSize();
        const float sx = client.x > 0 ? float(render.width) / float(client.x) : 1.0f;
        const float sy = client.y > 0 ? float(render.height) / float(client.y) : 1.0f;
        return float32x2(x * sx, y * sy);
    }

    ShadertoyPush buildPushConstants(const VkExtent2D& extent, double time) const
    {
        ShadertoyPush push {};
        push.time = float(time);
        push.resolution = float32x2(float(extent.width), float(extent.height));
        push.mouse = m_mouse;
        return push;
    }

    void dispatchPass(VkCommandBuffer cmd, const PassGpu& pass, const VkExtent2D& extent, double time)
    {
        RenderTarget* target = passTarget(pass.id);
        if (!target || !(*target) || pass.pipeline == VK_NULL_HANDLE || pass.set == VK_NULL_HANDLE)
        {
            return;
        }

        target->clear(cmd, float32x4(0.0f, 0.0f, 0.0f, 1.0f));

        const ShadertoyPush push = buildPushConstants(extent, time);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &pass.set, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        const u32 groupsX = (extent.width + 15) / 16;
        const u32 groupsY = (extent.height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    void encodeShader(VkCommandBuffer cmd, const VkExtent2D& extent, double time)
    {
        if (m_passes.empty() || !m_renderTarget || !(*m_renderTarget))
        {
            if (m_renderTarget && *m_renderTarget)
            {
                m_renderTarget->clear(cmd, float32x4(0.02f, 0.03f, 0.06f, 1.0f));
            }
            return;
        }

        for (size_t i = 0; i < m_passes.size(); ++i)
        {
            const PassGpu& pass = m_passes[i];
            dispatchPass(cmd, pass, extent, time);

            // After buffer writes, make them visible to later compute sampling.
            if (pass.id != PassId::Image)
            {
                RenderTarget* target = passTarget(pass.id);
                if (target && *target)
                {
                    VkImageMemoryBarrier barrier = makeImageBarrier(
                        target->image(),
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_GENERAL);

                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);
                }
            }
        }
    }

    void recordCommandBuffer(VkCommandBuffer cmd, u32 imageIndex, const VkExtent2D& extent, double time)
    {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo =
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(cmd, &beginInfo);

        encodeShader(cmd, extent, time);

        if (m_renderTarget && *m_renderTarget)
        {
            VkImageMemoryBarrier barrier = makeImageBarrier(
                m_renderTarget->image(),
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL);

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            m_renderTarget->resolve(cmd, swapchain(), imageIndex, &m_outputOptions);
        }

        vkEndCommandBuffer(cmd);
    }

    void render(double time)
    {
        auto frame = beginDraw();
        if (!frame || !m_renderTarget || !(*m_renderTarget))
        {
            return;
        }

        const VkExtent2D extent = swapchainExtent();
        VkCommandBuffer cmd = commandBuffer(frame.imageIndex());
        recordCommandBuffer(cmd, frame.imageIndex(), extent, time);
        frame.submitAndPresent(m_graphicsQueue, cmd);
    }

    void onFrame(const FrameInfo& info) override
    {
        m_frameTimeMs = float(info.dt * 1000.0);
        const float fps = 1000.0f / std::max(m_frameTimeMs, 0.001f);
        setTitle(fmt::format("{}  {:.1f} fps", m_statusLine, fps));
        render(info.time);
    }

    void onMouseMove(int x, int y) override
    {
        const float32x2 p = clientToRender(float(x), float(y));
        m_mouse.x = p.x;
        m_mouse.y = p.y;
        if (m_mouseDown)
        {
            m_mouse.z = p.x;
            m_mouse.w = p.y;
        }
    }

    void onMouseClick(int x, int y, MouseButton button, int count) override
    {
        if (button != MOUSEBUTTON_LEFT)
        {
            return;
        }

        const float32x2 p = clientToRender(float(x), float(y));
        m_mouse.x = p.x;
        m_mouse.y = p.y;

        if (count > 0)
        {
            m_mouseDown = true;
            m_mouse.z = p.x;
            m_mouse.w = p.y;
        }
        else
        {
            m_mouseDown = false;
            m_mouse.z = 0.0f;
            m_mouse.w = 0.0f;
        }
    }

    void onKeyPress(Keycode code, u32 mask) override
    {
        MANGO_UNREFERENCED(mask);

        if (code == KEYCODE_ESC)
        {
            breakEventLoop();
        }
        else if (code == KEYCODE_F)
        {
            toggleFullscreen();
        }
        else if (code == KEYCODE_LEFT)
        {
            cycleShader(-1);
        }
        else if (code == KEYCODE_RIGHT)
        {
            cycleShader(1);
        }
    }
};

int mangoMain(const mango::CommandLine& commands)
{
    std::vector<const char*> enabledLayers;
    bool validation = false;
    bool forceSdr = false;
    std::string initialEffect;

    for (size_t i = 1; i < commands.size(); ++i)
    {
        const std::string arg = std::string(commands[i]);
        if (arg == "--info")
        {
            printEnable(Print::Info, true);
        }
        else if (arg == "--validate")
        {
            validation = true;
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        }
        else if (arg == "--sdr")
        {
            forceSdr = true;
        }
        else if (arg == "--shader" && i + 1 < commands.size())
        {
            initialEffect = std::string(commands[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            printLine("shadertoy  [<file.comp>|--shader <file>] [--info] [--validate] [--sdr]");
            printLine("  <file.comp>  initial effect (name or path; basename matched)");
            printLine("  --shader     same as positional file argument");
            printLine("  --info       log device/swapchain details");
            printLine("  --validate   enable Khronos validation layer + debug messenger");
            printLine("  --sdr        force SDR swapchain (tonemap fallback)");
            printLine("  LEFT/RIGHT   cycle through shaders");
            printLine("  MOUSE        mouse movement is used by some shaders");
            return 0;
        }
        else if (!arg.empty() && arg[0] != '-')
        {
            initialEffect = arg;
        }
    }

    Path path("data/");

    const ShaderLibrary library = loadCompiledShaders(path);
    if (library.effects.empty())
    {
        printLine(Print::Error, "No shaders found in {}", path.pathname());
        return 1;
    }

    size_t initialIndex = 0;
    if (!initialEffect.empty())
    {
        const std::string key = effectKeyFromArg(initialEffect);
        const size_t found = findEffectIndex(library, key);
        if (found == size_t(-1))
        {
            printLine(Print::Error, "Effect not found: {} (looked for {})", initialEffect, key);
            return 1;
        }
        initialIndex = found;
        printLine("Starting effect [{}/{}]: {}", initialIndex + 1, library.effects.size(), key);
    }

    if (validation)
    {
        printEnable(Print::Warning, true);
    }

    std::vector<const char*> enabledExtensions = vulkan::requiredSurfaceExtensions();

    InstanceExtensionProperties instanceExtensionProperties;
    if (instanceExtensionProperties.contains(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME))
    {
        enabledExtensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }

    VkApplicationInfo applicationInfo =
    {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "shadertoy",
        .applicationVersion = 1,
        .pEngineName = "mango",
        .engineVersion = 1,
        .apiVersion = VK_MAKE_VERSION(1, 3, 0),
    };

    Instance instance(applicationInfo, enabledLayers, enabledExtensions);

    VulkanDeviceConfig deviceConfig {};
    applyRecommendedSurfaceFormats(deviceConfig, forceSdr ? SurfaceFormatIntent::SDR : SurfaceFormatIntent::HDR);

    ShadertoyWindow window(instance, 1280, 720, &deviceConfig, library, initialIndex);

    EventLoopConfig config;
    config.mode = FrameMode::Continuous;
    config.waitForFrame = true;

    window.enterEventLoop(config);

    return 0;
}
