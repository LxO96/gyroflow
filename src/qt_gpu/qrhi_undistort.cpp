// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2021-2022 Adrian <adrian.eddy at gmail>

#include <QQuickWindow>
#include <QFile>
#include <private/qquickitem_p.h>
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
#   include <rhi/qrhi.h>
#else
#   include <private/qrhi_p.h>
#endif
#include <private/qsgrenderer_p.h>
#include <private/qsgdefaultrendercontext_p.h>
#include <private/qshader_p.h>

#define qDebug2(func) QMessageLogger(__FILE__, __LINE__, func).debug(QLoggingCategory("Qt RHI"))

class MDKPlayer {
public:
    QSGDefaultRenderContext *rhiContext();
    QRhiTexture *rhiTexture();
    // QRhiTextureRenderTarget *rhiRenderTarget();
    // QRhiRenderPassDescriptor *rhiRenderPassDescriptor();
    QQuickWindow *qmlWindow();
    QQuickItem *qmlItem();
    QSize textureSize();
    QMatrix4x4 textureMatrix();

    void *userData() const;
    void setUserData(void *);
    void setUserDataDestructor(std::function<void(void *)> &&cb);
};
class MDKPlayerWrapper {
public:
    MDKPlayer *mdkplayer;
};

static float quadVertexData[16] = { // Y up, CCW
    -0.5f,  0.5f, 0.0f, 0.0f,
    -0.5f, -0.5f, 0.0f, 1.0f,
    0.5f, -0.5f, 1.0f, 1.0f,
    0.5f,  0.5f, 1.0f, 0.0f
};
static quint16 quadIndexData[6] = { 0, 1, 2, 0, 2, 3 };

// 2³ identity LUT in .cube order (RGB-major, red varying fastest), used until a real LUT
// is loaded so that binding 6 is never left unbound - that is a validation error on
// several RHI backends.
static const float identityLut2[8 * 3] = {
    0.0f, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,   1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f,   1.0f, 1.0f, 1.0f
};

// ubufAlignment
// static inline uint aligned(uint v, uint byteAlign) { return (v + byteAlign - 1) & ~(byteAlign - 1); }

class QtRHIUndistort {
public:
    QtRHIUndistort() { }

    QSize outSize() { return m_outputSize; }
    QSize texSize() { return m_textureSize; }
    QString shaderPath() { return m_shaderPath; }
    QRhiTexture *itemTexturePtr() { return m_itemTexturePtr; }
    unsigned int sizeForRS() { return m_sizeForRS; }

    bool init(MDKPlayer *item, QSize textureSize, QSize outputSize, const QString &shaderPath, int kernelParmsSize, unsigned int sizeForRS, QSize canvasSize) {
        if (!item) return false;
        auto context = item->rhiContext();
        auto rhi = context->rhi();

        m_sizeForRS = sizeForRS;
        m_outputSize = outputSize;
        m_textureSize = textureSize;
        m_shaderPath = shaderPath;
        m_itemTexturePtr = item->rhiTexture();

        m_texIn.reset(rhi->newTexture(QRhiTexture::RGBA8, textureSize, 1, QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
        if (!m_texIn->create()) { qDebug2("init") << "failed to create m_texIn"; return false; }

        m_rt.reset(rhi->newTextureRenderTarget({ QRhiColorAttachment(m_texIn.get()) }));
        if (!m_rt) { qDebug2("init") << "failed to get new m_rt"; return false; }

        m_rtRp.reset(m_rt->newCompatibleRenderPassDescriptor());
        if (!m_rtRp) { qDebug2("init") << "failed to create m_rtRp"; return false; }

        m_rt->setRenderPassDescriptor(m_rtRp.get());
        if (!m_rt->create()) { qDebug2("init") << "failed to create m_rt"; return false; }

        m_kernelParams.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, kernelParmsSize));
        if (!m_kernelParams->create()) { qDebug2("init") << "failed to create m_kernelParams"; return false; }

        m_texMatrices.reset(rhi->newTexture(QRhiTexture::R32F, QSize(14, sizeForRS), 1, QRhiTexture::Flags()));
        if (!m_texMatrices->create()) { qDebug2("init") << "failed to create m_texMatrices"; return false; }

        m_texMeshData.reset(rhi->newTexture(QRhiTexture::R32F, QSize(1, 1024), 1, QRhiTexture::Flags()));
        if (!m_texMeshData->create()) { qDebug2("init") << "failed to create m_texMeshData"; return false; }

        matricesBuffer.resize(sizeForRS * 14 * sizeof(float));
        meshDataBuffer.resize(1024 * sizeof(float));

        m_texCanvas.reset(rhi->newTexture(QRhiTexture::R8, canvasSize, 1, QRhiTexture::Flags()));
        if (!m_texCanvas->create()) { qDebug2("init") << "failed to create m_texCanvas"; return false; }

        m_vertexBuffer.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, sizeof(quadVertexData)));
        if (!m_vertexBuffer->create()) { qDebug2("init") << "failed to create m_vertexBuffer"; return false; }

        m_indexBuffer.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, sizeof(quadIndexData)));
        if (!m_indexBuffer->create()) { qDebug2("init") << "failed to create m_indexBuffer"; return false; }

        m_drawingUniform.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 64 + 4));
        if (!m_drawingUniform->create()) { qDebug2("init") << "failed to create m_drawingUniform"; return false; }
        qint32 flip = rhi->isYUpInFramebuffer();

        m_drawingSampler.reset(rhi->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!m_drawingSampler->create()) { qDebug2("init") << "failed to create m_drawingSampler"; return false; }

        m_canvasSampler.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!m_canvasSampler->create()) { qDebug2("init") << "failed to create m_canvasSampler"; return false; }

        m_matricesSampler.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!m_matricesSampler->create()) { qDebug2("init") << "failed to create m_matricesSampler"; return false; }

        m_meshDataSampler.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!m_meshDataSampler->create()) { qDebug2("init") << "failed to create m_meshDataSampler"; return false; }

        // The 3D LUT is tiled into 2D: (size * size) wide by size tall. The fragment shader
        // does its own trilinear interpolation via texelFetch, so this sampler MUST stay
        // Nearest in both filters with ClampToEdge addressing - any hardware filtering would
        // make the preview diverge from the export.
        m_lutSampler.reset(rhi->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        if (!m_lutSampler->create()) { qDebug2("init") << "failed to create m_lutSampler"; return false; }

        // Format ladder for the LUT texture. Losing the LUT must never cost the user the
        // whole preview, so this never fails init():
        //   rung 0 - RGBA32F: preferred, matches the f32 precision of the other backends.
        //   rung 1 - RGBA16F: real LUTs still work, at ~11-bit mantissa precision. Logged,
        //            because it widens the cross-backend tolerance.
        //   rung 2 - RGBA8:   the 2 identity only; real LUT data is refused. 0.0 and 1.0
        //            are exactly representable in 8 bits, so lut_sample() returns its input
        //            and the LUT becomes a precise no-op. Exposure, contrast, saturation and
        //            white balance all still work.
        if (rhi->isTextureFormatSupported(QRhiTexture::RGBA32F)) {
            m_lutFormat = QRhiTexture::RGBA32F;
            m_lutRung = 0;
        } else if (rhi->isTextureFormatSupported(QRhiTexture::RGBA16F)) {
            m_lutFormat = QRhiTexture::RGBA16F;
            m_lutRung = 1;
            qDebug2("init") << "QRhiTexture::RGBA32F is not supported by this RHI backend; the preview LUT falls back to RGBA16F (~11-bit mantissa), so the preview may differ slightly from the export";
        } else {
            m_lutFormat = QRhiTexture::RGBA8;
            m_lutRung = 2;
            qDebug2("init") << "neither QRhiTexture::RGBA32F nor RGBA16F is supported by this RHI backend; colour LUTs are unavailable in the preview (every other colour adjustment still applies)";
        }
        // Always created, even with no LUT loaded, so binding 6 is never left unbound.
        if (!createLutTexture(rhi, 2) && m_lutFormat != QRhiTexture::RGBA8) {
            // Not a format-support problem, so drop to the universally supported format
            // rather than giving up on the binding.
            qDebug2("init") << "failed to create the preview LUT texture; falling back to RGBA8 identity";
            m_lutFormat = QRhiTexture::RGBA8;
            m_lutRung = 2;
            createLutTexture(rhi, 2);
        }
        // A 4x2 RGBA8 texture failing means the device is out of resources entirely; there is
        // no way to build a valid SRB from here, and drawing with binding 6 unbound is a
        // validation error on several backends. This is the one LUT failure that stops init.
        if (!m_texLut) { qDebug2("init") << "failed to create m_texLut in any supported format"; return false; }

        if (!updateBindings(rhi)) { qDebug2("init") << "failed to create m_srb"; return false; }

        m_pipeline.reset(rhi->newGraphicsPipeline());
        m_pipeline->setShaderStages({
            { QRhiShaderStage::Vertex,   getShader(QLatin1String(":/src/qt_gpu/compiled/texture.vert.qsb")) },
            { QRhiShaderStage::Fragment, getShader(shaderPath) }
        });
        QRhiVertexInputLayout inputLayout;
        inputLayout.setBindings({ { 4 * sizeof(float) } });
        inputLayout.setAttributes({
            { 0, 0, QRhiVertexInputAttribute::Float2, 0 },
            { 0, 1, QRhiVertexInputAttribute::Float2, 2 * sizeof(float) }
        });
        m_pipeline->setVertexInputLayout(inputLayout);
        m_pipeline->setShaderResourceBindings(m_srb.get());
        m_pipeline->setRenderPassDescriptor(m_rtRp.get());
        if (!m_pipeline->create()) { qDebug2("init") << "failed to create m_pipeline"; return false; }

        m_initialUpdates = rhi->nextResourceUpdateBatch();
        m_initialUpdates->uploadStaticBuffer(m_vertexBuffer.get(), quadVertexData);
        m_initialUpdates->uploadStaticBuffer(m_indexBuffer.get(), quadIndexData);
        m_initialUpdates->updateDynamicBuffer(m_drawingUniform.get(), 64, 4, &flip);
        uploadLut(m_initialUpdates, identityLut2, 2, true);

        return true;
    }

    // The binding layout never changes, so recreating m_srb after swapping the LUT texture
    // leaves m_pipeline valid (QRhi only requires layout compatibility).
    //
    // On failure m_bindingsValid latches false: setBindings() has already mutated the
    // object, so a failed create() leaves the SRB in an indeterminate state and nothing may
    // draw with it. render() checks the flag and refuses the pass.
    bool updateBindings(QRhi *rhi) {
        if (!m_srb) m_srb.reset(rhi->newShaderResourceBindings());
        m_srb->setBindings({
            QRhiShaderResourceBinding::uniformBuffer (0, QRhiShaderResourceBinding::FragmentStage | QRhiShaderResourceBinding::VertexStage, m_drawingUniform.get()),
            QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, m_itemTexturePtr, m_drawingSampler.get()),
            QRhiShaderResourceBinding::uniformBuffer (2, QRhiShaderResourceBinding::FragmentStage, m_kernelParams.get()),
            QRhiShaderResourceBinding::sampledTexture(3, QRhiShaderResourceBinding::FragmentStage, m_texMatrices.get(), m_matricesSampler.get()),
            QRhiShaderResourceBinding::sampledTexture(4, QRhiShaderResourceBinding::FragmentStage, m_texCanvas.get(), m_canvasSampler.get()),
            QRhiShaderResourceBinding::sampledTexture(5, QRhiShaderResourceBinding::FragmentStage, m_texMeshData.get(), m_meshDataSampler.get()),
            QRhiShaderResourceBinding::sampledTexture(6, QRhiShaderResourceBinding::FragmentStage, m_texLut.get(), m_lutSampler.get()),
        });
        const bool ok = m_srb->create();
        if (!ok) m_bindingsValid = false;
        return ok;
    }

    // Builds into a local and only publishes it on success, so a failed create() can never
    // leave m_texLut (and therefore m_srb) pointing at a destroyed texture.
    bool createLutTexture(QRhi *rhi, uint32_t size) {
        QScopedPointer<QRhiTexture> tex(rhi->newTexture(m_lutFormat, QSize(int(size * size), int(size)), 1, QRhiTexture::Flags()));
        if (!tex || !tex->create()) { qDebug2("createLutTexture") << "failed to create m_texLut of size" << size; return false; }
        m_texLut.reset(tex.take());
        m_lutSize = size;
        return true;
    }

    // Swap in a differently sized LUT texture and rebind it. Either both succeed, or the old
    // texture and the old bindings are left exactly as they were - the SRB must never be left
    // referencing freed memory, because the caller goes on to draw.
    bool resizeLutTexture(QRhi *rhi, uint32_t size) {
        QScopedPointer<QRhiTexture> tex(rhi->newTexture(m_lutFormat, QSize(int(size * size), int(size)), 1, QRhiTexture::Flags()));
        if (!tex || !tex->create()) {
            // The checksum is deliberately not advanced by the caller, so the next frame
            // retries - which is what makes this self-healing. The log line is latched
            // instead, otherwise a persistent condition (device OOM with a large LUT) would
            // print once per frame forever. Reset on success so a later failure is heard.
            if (!m_lutResizeFailureLogged) {
                qDebug2("resizeLutTexture") << "failed to create the LUT texture at size" << size << "- keeping the previous one, and retrying on later frames";
                m_lutResizeFailureLogged = true;
            }
            return false;
        }

        QRhiTexture *previous = m_texLut.take(); // Kept alive until the new bindings are live.
        const uint32_t previousSize = m_lutSize;
        m_texLut.reset(tex.take());
        m_lutSize = size;

        if (!updateBindings(rhi)) {
            qDebug2("resizeLutTexture") << "failed to rebind the resized LUT texture - restoring the previous one";
            m_texLut.reset(previous);
            m_lutSize = previousSize;
            // m_bindingsValid was latched false by the failed updateBindings above. If the
            // restore succeeds the SRB is coherent again, so clear it; if it fails too,
            // render() will refuse to draw.
            m_bindingsValid = true;
            if (!updateBindings(rhi)) qDebug2("resizeLutTexture") << "failed to restore the previous bindings; the preview will stop rendering";
            return false;
        }

        delete previous;
        m_lutResizeFailureLogged = false;
        return true;
    }

    // `data` is `size³ * 3` floats in .cube order (RGB-major, red varying fastest), which is
    // exactly the row-major order of the (size*size) x size tiled image the shader reads.
    // `isIdentity` says whether `data` is the 2 identity, which is the only content the
    // RGBA8 rung can represent exactly.
    //
    // Returns false when nothing was uploaded, so the caller must not advance m_lutChecksum.
    bool uploadLut(QRhiResourceUpdateBatch *u, const float *data, uint32_t size, bool isIdentity) {
        if (!m_texLut) return false;
        const int texels = int(size * size * size);
        QByteArray rgba;
        if (m_lutFormat == QRhiTexture::RGBA32F) {
            // Expand the RGB triplets to 4 components.
            rgba.resize(texels * 4 * int(sizeof(float)));
            float *dst = reinterpret_cast<float *>(rgba.data());
            for (int i = 0; i < texels; ++i) {
                dst[i * 4 + 0] = data[i * 3 + 0];
                dst[i * 4 + 1] = data[i * 3 + 1];
                dst[i * 4 + 2] = data[i * 3 + 2];
                dst[i * 4 + 3] = 1.0f;
            }
        } else if (m_lutFormat == QRhiTexture::RGBA16F) {
            rgba.resize(texels * 4 * int(sizeof(qfloat16)));
            qfloat16 *dst = reinterpret_cast<qfloat16 *>(rgba.data());
            for (int i = 0; i < texels; ++i) {
                dst[i * 4 + 0] = qfloat16(data[i * 3 + 0]);
                dst[i * 4 + 1] = qfloat16(data[i * 3 + 1]);
                dst[i * 4 + 2] = qfloat16(data[i * 3 + 2]);
                dst[i * 4 + 3] = qfloat16(1.0f);
            }
        } else {
            // RGBA8 can only hold the identity exactly (0.0 and 1.0 round-trip); anything
            // else would grade the preview differently from the export, so refuse it.
            if (!isIdentity) {
                if (!m_lutRefusalLogged) {
                    qDebug2("uploadLut") << "refusing to upload a colour LUT: this RHI backend supports neither RGBA32F nor RGBA16F, so the preview keeps an identity LUT (every other colour adjustment still applies)";
                    m_lutRefusalLogged = true;
                }
                return false;
            }
            rgba.resize(texels * 4);
            uchar *dst = reinterpret_cast<uchar *>(rgba.data());
            for (int i = 0; i < texels; ++i) {
                for (int c = 0; c < 3; ++c) {
                    const float v = qBound(0.0f, data[i * 3 + c], 1.0f);
                    dst[i * 4 + c] = uchar(v * 255.0f + 0.5f);
                }
                dst[i * 4 + 3] = 255;
            }
        }
        QRhiTextureSubresourceUploadDescription desc(rgba.constData(), rgba.size());
        u->uploadTexture(m_texLut.get(), QRhiTextureUploadDescription({ QRhiTextureUploadEntry(0, 0, desc) }));
        return true;
    }

    bool render(MDKPlayer *item, uint8_t *params, uint paramsLen, uint8_t *matrices, uint matricesLen, uint8_t *canvas, uint canvasLen, float *meshData, uint meshDataLen, const float *lutData, uint32_t lutSize, uint32_t lutChecksum) {
        if (!item->qmlItem() || !item->rhiTexture() || !item->qmlWindow()) return false;
        // A failed QRhiShaderResourceBindings::create() leaves the object indeterminate;
        // drawing with it is undefined, so refuse the frame rather than risk it.
        if (!m_bindingsValid) return false;
        auto context = item->rhiContext();
        auto rhi = context->rhi();

        if (matricesBuffer.size() < matricesLen) { matricesBuffer.resize(matricesLen); }
        if (matricesLen > 0) memcpy(matricesBuffer.data(), matrices, matricesLen);

        if (meshDataBuffer.size() < meshDataLen*4) { meshDataBuffer.resize(meshDataLen*4); }
        if (meshDataLen > 0) memcpy(meshDataBuffer.data(), meshData, meshDataLen*4);
        else if (meshDataBuffer[0] != 0) memset(meshDataBuffer.data(), 0, meshDataBuffer.size());

        const QSize size = item->textureSize();
        QRhiCommandBuffer *cb = context->currentFrameCommandBuffer();

        QRhiResourceUpdateBatch *u = rhi->nextResourceUpdateBatch();
        if (m_initialUpdates) {
            u->merge(m_initialUpdates);
            m_initialUpdates->release();
            m_initialUpdates = nullptr;
        }

        u->updateDynamicBuffer(m_kernelParams.get(), 0, paramsLen, params);

        {
            QRhiTextureSubresourceUploadDescription desc1(meshDataBuffer.data(), meshDataBuffer.size());
            u->uploadTexture(m_texMeshData.get(), QRhiTextureUploadDescription({ QRhiTextureUploadEntry(0, 0, desc1) }));
        }

        QRhiTextureSubresourceUploadDescription desc1(matricesBuffer.data(), matricesBuffer.size());
        u->uploadTexture(m_texMatrices.get(), QRhiTextureUploadDescription({ QRhiTextureUploadEntry(0, 0, desc1) }));

        if (canvasLen > 0) {
            QRhiTextureSubresourceUploadDescription desc2(canvas, canvasLen);
            u->uploadTexture(m_texCanvas.get(), QRhiTextureUploadDescription({ QRhiTextureUploadEntry(0, 0, desc2) }));
        }

        // Re-upload the LUT only when it actually changed. The checksum is the crc32 of the
        // LUT data from `Lut3D`, so 0 means "no LUT loaded" and keeps the identity uploaded
        // by init().
        // A resize that fails leaves the previous texture and bindings intact, and the
        // checksum unadvanced, so the next frame simply tries again.
        if (lutData && lutSize >= 2 && lutChecksum != m_lutChecksum) {
            if (m_lutRung == 2) {
                // No float format on this backend: the identity stays and there is nothing
                // to retry, so record the checksum to avoid re-deciding every frame.
                uploadLut(u, lutData, lutSize, false); // refuses, and logs once
                m_lutChecksum = lutChecksum;
            } else if (lutSize == m_lutSize || resizeLutTexture(rhi, lutSize)) {
                if (uploadLut(u, lutData, lutSize, false)) m_lutChecksum = lutChecksum;
            }
        }
        // resizeLutTexture() runs mid-frame and can latch m_bindingsValid false when its
        // restore rebind also fails, so the check at the top of render() is not enough -
        // re-check here, before anything draws with the SRB.
        //
        // `u` has been acquired and written to by this point. Update batches come from a
        // fixed-size pool and are only returned to it by beginPass()/resourceUpdate() or by
        // an explicit release(), so bailing out without releasing would consume a pool slot
        // for good. This is the only early return in render() that sits after
        // nextResourceUpdateBatch().
        if (!m_bindingsValid) { u->release(); return false; }

        QMatrix4x4 mvp = item->textureMatrix();
        mvp.scale(2.0f);
        u->updateDynamicBuffer(m_drawingUniform.get(), 0, 64, mvp.constData());

        cb->beginPass(m_rt.get(), QColor(Qt::black), { 1.0f, 0 }, u);
        cb->setGraphicsPipeline(m_pipeline.get());
        cb->setViewport({ 0, 0, float(size.width()), float(size.height()) });
        cb->setShaderResources();
        QRhiCommandBuffer::VertexInput vbufBinding(m_vertexBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbufBinding, m_indexBuffer.get(), 0, QRhiCommandBuffer::IndexUInt16);
        cb->drawIndexed(6);
        cb->endPass();

        u = rhi->nextResourceUpdateBatch();
        u->copyTexture(item->rhiTexture(), m_texIn.get(), {});
        cb->resourceUpdate(u);

        rhi->finish();

        return true;
    }

    std::vector<uint8_t> matricesBuffer;
    std::vector<uint8_t> meshDataBuffer;

    /*QByteArray getContents(const QString &name) {
        QFile f(name);
        if (f.open(QIODevice::ReadOnly))
            return f.readAll();
        return QByteArray();
    }*/
    QShader getShader(const QString &name) {
        QFile f(name);
        if (f.open(QIODevice::ReadOnly))
            return QShader::fromSerialized(f.readAll());
        return QShader();
    }

    QRhiTexture *m_itemTexturePtr{nullptr};

    QScopedPointer<QRhiTexture> m_texIn;
    QScopedPointer<QRhiTexture> m_texMatrices;
    QScopedPointer<QRhiTexture> m_texCanvas;
    QScopedPointer<QRhiBuffer> m_kernelParams;
    QScopedPointer<QRhiTexture> m_texMeshData;
    QScopedPointer<QRhiTexture> m_texLut;
    uint32_t m_lutSize{0};
    uint32_t m_lutChecksum{0};
    // Which rung of the LUT format ladder init() settled on: 0 = RGBA32F, 1 = RGBA16F,
    // 2 = RGBA8 identity only (LUTs unavailable in the preview on this backend).
    QRhiTexture::Format m_lutFormat{QRhiTexture::RGBA32F};
    int m_lutRung{0};
    bool m_lutRefusalLogged{false};
    bool m_lutResizeFailureLogged{false};
    bool m_bindingsValid{true};

    QSize m_outputSize;
    QSize m_textureSize;
    QString m_shaderPath;
    unsigned int m_sizeForRS{0};

    QScopedPointer<QRhiBuffer> m_vertexBuffer;
    QScopedPointer<QRhiBuffer> m_indexBuffer;
    QScopedPointer<QRhiBuffer> m_drawingUniform;
    QScopedPointer<QRhiSampler> m_canvasSampler;
    QScopedPointer<QRhiSampler> m_drawingSampler;
    QScopedPointer<QRhiSampler> m_matricesSampler;
    QScopedPointer<QRhiSampler> m_meshDataSampler;
    QScopedPointer<QRhiSampler> m_lutSampler;
    QScopedPointer<QRhiShaderResourceBindings> m_srb;
    QScopedPointer<QRhiGraphicsPipeline> m_pipeline;

    QScopedPointer<QRhiTextureRenderTarget> m_rt;
    QScopedPointer<QRhiRenderPassDescriptor> m_rtRp;

    QScopedPointer<QRhiReadbackResult> m_readbackResult;

    QRhiResourceUpdateBatch *m_initialUpdates{nullptr};
};
