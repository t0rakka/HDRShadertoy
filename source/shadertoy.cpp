/*
    MANGO Multimedia Development Platform
    Copyright (C) 2012-2026 Twilight Finland 3D Oy Ltd. All rights reserved.

    Vulkan Shadertoy compute shader player — Float16 RenderTarget, HDR resolve.
*/
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
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

    struct CompiledShader
    {
        size_t spirvOffset = 0;
        size_t spirvCount = 0;
        std::string name;

        std::span<const u32> view(const std::vector<u32>& spirv) const
        {
            return { spirv.data() + spirvOffset, spirvCount };
        }
    };

    struct ShaderLibrary
    {
        std::vector<u32> spirv;
        std::vector<CompiledShader> shaders;
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

    std::string basename(const std::string& path)
    {
        const size_t slash = path.find_last_of("/\\");
        return slash == std::string::npos ? path : path.substr(slash + 1);
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
        std::vector<std::pair<std::vector<u32>, std::string>> compiled;
        compiled.reserve(names.size());

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
            Shader shader = compiler.compile(source, ShaderStage::Compute);
            if (!shader)
            {
                printLine(Print::Error, "Compute shader compile failed: {}", name);
                if (!shader.log.empty())
                {
                    printLine(Print::Error, "{}", shader.log);
                }
                continue;
            }

            compiled.emplace_back(shader.spirv, name);
        }

        ShaderLibrary library;
        size_t spirvWords = 0;
        for (const auto& [spirv, name] : compiled)
        {
            MANGO_UNREFERENCED(name);
            spirvWords += spirv.size();
        }

        library.spirv.reserve(spirvWords);

        for (auto& [spirv, name] : compiled)
        {
            const size_t offset = library.spirv.size();
            library.spirv.insert(library.spirv.end(), spirv.begin(), spirv.end());
            library.shaders.push_back({
                .spirvOffset = offset,
                .spirvCount = spirv.size(),
                .name = std::move(name),
            });
        }

        return library;
    }

    size_t findShaderIndex(const ShaderLibrary& library, const std::string& name)
    {
        for (size_t i = 0; i < library.shaders.size(); ++i)
        {
            if (library.shaders[i].name == name)
            {
                return i;
            }
        }

        return 0;
    }

} // namespace

class ShadertoyWindow : public VulkanWindow
{
protected:
    std::unique_ptr<Allocator> m_allocator;
    std::unique_ptr<RenderTarget> m_renderTarget;

    OutputTransformOptions m_outputOptions {};
    bool m_hdrSwapchain = false;

    ShaderLibrary m_library;
    size_t m_shaderIndex = 0;

    VkShaderModule m_computeShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    float m_frameTimeMs = 0.0f;
    std::string m_statusLine;

    float32x4 m_mouse {};
    bool m_mouseDown = false;

public:
    ShadertoyWindow(VkInstance instance, int width, int height, const VulkanDeviceConfig* config,
                     ShaderLibrary library, size_t initialIndex = 0)
        : VulkanWindow(instance, width, height, 0, config)
        , m_library(std::move(library))
        , m_shaderIndex(std::min(initialIndex, m_library.shaders.empty() ? 0 : m_library.shaders.size() - 1))
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
        applyShader(m_shaderIndex);
        recreateRenderTarget(swapchainExtent());
        updateStatusLine();
    }

    void onSwapchainResize(VkExtent2D extent) override
    {
        m_hdrSwapchain = isHDR(surfaceFormat());
        m_outputOptions = defaultOutputOptions(surfaceFormat(), m_outputOptions.exposure);
        recreateRenderTarget(extent);
        updateStatusLine();
    }

    ~ShadertoyWindow()
    {
        if (m_device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device);
            destroyComputeResources();
            m_renderTarget.reset();
            m_allocator.reset();
        }
    }

    void createComputeResources()
    {
        if (m_library.shaders.empty())
        {
            printLine(Print::Error, "No compiled shaders available.");
            return;
        }

        VkDescriptorSetLayoutBinding binding =
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding,
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

        VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
        VkDescriptorPoolCreateInfo poolInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };

        vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);
    }

    void applyShader(size_t index)
    {
        if (m_library.shaders.empty() || index >= m_library.shaders.size() || !m_pipelineLayout)
        {
            return;
        }

        vkDeviceWaitIdle(m_device);

        if (m_pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }

        if (m_computeShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(m_device, m_computeShader, nullptr);
            m_computeShader = VK_NULL_HANDLE;
        }

        m_shaderIndex = index;

        const CompiledShader& compiled = m_library.shaders[m_shaderIndex];
        const std::span<const u32> spirvSpan = compiled.view(m_library.spirv);
        if (spirvSpan.empty())
        {
            printLine(Print::Error, "SPIR-V data is empty: {}", compiled.name);
            return;
        }

        const std::vector<u32> spirv(spirvSpan.begin(), spirvSpan.end());
        m_computeShader = Compiler::createShaderModule(m_device, spirv);

        VkPipelineShaderStageCreateInfo stage =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = m_computeShader,
            .pName = "main",
        };

        VkComputePipelineCreateInfo pipelineInfo =
        {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stage,
            .layout = m_pipelineLayout,
        };

        vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);

        if (m_pipeline == VK_NULL_HANDLE)
        {
            printLine(Print::Error, "vkCreateComputePipelines failed: {}", compiled.name);
            return;
        }

        printLine(Print::Info, "Shader [{}/{}]: {}", m_shaderIndex + 1, m_library.shaders.size(), compiled.name);
        updateStatusLine();
    }

    void cycleShader(int delta)
    {
        if (m_library.shaders.size() < 2)
        {
            return;
        }

        const size_t count = m_library.shaders.size();
        const size_t next = (m_shaderIndex + count + size_t(delta)) % count;
        applyShader(next);
    }

    void destroyComputeResources()
    {
        if (m_descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }

        if (m_pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
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

        if (m_computeShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(m_device, m_computeShader, nullptr);
            m_computeShader = VK_NULL_HANDLE;
        }

        m_descriptorSet = VK_NULL_HANDLE;
    }

    void updateTargetDescriptor()
    {
        if (!m_renderTarget || !(*m_renderTarget) || !m_descriptorPool)
        {
            return;
        }

        if (m_descriptorSet == VK_NULL_HANDLE)
        {
            VkDescriptorSetAllocateInfo allocInfo =
            {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = m_descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &m_descriptorLayout,
            };

            VkResult result = vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet);
            if (result != VK_SUCCESS)
            {
                printLine(Print::Error, "vkAllocateDescriptorSets: {}", getString(result));
                m_descriptorSet = VK_NULL_HANDLE;
                return;
            }
        }

        VkDescriptorImageInfo imageInfo =
        {
            .imageView = m_renderTarget->view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        VkWriteDescriptorSet write =
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo,
        };

        vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    }

    void recreateRenderTarget(VkExtent2D extent)
    {
        if (!m_allocator || extent.width == 0 || extent.height == 0)
        {
            return;
        }

        m_renderTarget = std::make_unique<RenderTarget>(RenderTarget::CreateInfo
        {
            .device = m_device,
            .allocator = m_allocator.get(),
            .queue = m_graphicsQueue,
            .queueFamily = m_graphicsQueueFamilyIndex,
            .format = RenderTargetFormat::Float16,
            .extent = extent,
        });

        updateTargetDescriptor();
    }

    void updateStatusLine()
    {
        const VkExtent2D render = swapchainExtent();
        const char* hdrLabel = m_hdrSwapchain ? "HDR" : "SDR (tonemap fallback)";
        const std::string shaderName = m_library.shaders.empty() ? "none" : m_library.shaders[m_shaderIndex].name;

        m_statusLine = fmt::format("{}  {}x{}  {}  Float16 RT  [{}/{}]",
            shaderName, render.width, render.height, hdrLabel,
            m_shaderIndex + 1, m_library.shaders.size());
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

    void encodeShader(VkCommandBuffer cmd, const VkExtent2D& extent, double time)
    {
        if (!m_pipeline || !m_descriptorSet)
        {
            m_renderTarget->clear(cmd, float32x4(0.02f, 0.03f, 0.06f, 1.0f));
            return;
        }

        m_renderTarget->clear(cmd, float32x4(0.0f, 0.0f, 0.0f, 1.0f));

        const ShadertoyPush push = buildPushConstants(extent, time);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        const u32 groupsX = (extent.width + 15) / 16;
        const u32 groupsY = (extent.height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
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
            VkImageMemoryBarrier barrier =
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = m_renderTarget->image(),
                .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
            };

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        m_renderTarget->resolve(cmd, swapchain(), imageIndex, &m_outputOptions);

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
        else if (arg == "--help" || arg == "-h")
        {
            printLine("shadertoy  [--shader <file.comp>] [--info] [--validate] [--sdr]");
            printLine("  --info       log device/swapchain details");
            printLine("  --validate   enable Khronos validation layer + debug messenger");
            printLine("  --sdr        force SDR swapchain (tonemap fallback)");
            printLine("  LEFT/RIGHT   cycle through shaders");
            printLine("  MOUSE        mouse movement is used by some shaders");
            return 0;
        }
    }

    Path path("data/");

    const ShaderLibrary library = loadCompiledShaders(path);
    if (library.shaders.empty())
    {
        printLine(Print::Error, "No shaders found in {}", path.pathname());
        return 1;
    }

    size_t initialIndex = 0;

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
