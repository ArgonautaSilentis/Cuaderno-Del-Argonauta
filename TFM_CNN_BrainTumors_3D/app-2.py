# app.py
import os
import hashlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import SimpleITK as sitk

# Instalar dependencias
import subprocess
import sys

from matplotlib.patches import Patch

# Diccionario de colores por clase
class_colors = {
    1: (1.0, 0.0, 0.0),  # rojo (NETC)
    2: (0.0, 1.0, 0.0),  # verde (SNFH)
    3: (1.0, 1.0, 0.0),  # amarillo (ET)
    4: (0.0, 1.0, 1.0),  # cyan (RC)
}

# Elementos de la leyenda para matplotlib
legend_elements = [
    Patch(facecolor=class_colors[1], label="1 - NETC"),
    Patch(facecolor=class_colors[2], label="2 - SNFH"),
    Patch(facecolor=class_colors[3], label="3 - ET"),
    Patch(facecolor=class_colors[4], label="4 - RC"),
]


### Preprocesado###
def bias_correct_image(nib_img, mask=None, shrink_factor=4, its=(50, 50, 50, 30)):
    sitk_img = sitk.GetImageFromArray(nib_img.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray((mask > 0).astype(np.uint8)) if mask is not None else None
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(its))
    result = corrector.Execute(sitk_img, mask_sitk) if mask_sitk else corrector.Execute(sitk_img)
    return sitk.GetArrayFromImage(result)

import tempfile
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

def preprocess_uploaded_modalities(nii_dict, target_size=128):
    # Crear carpeta temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        corrected_channels = []

        for m in ["t1n", "t1c", "t2f", "t2w"]:
            file_obj = nii_dict[m]

            # Guardar archivo temporal
            temp_path = os.path.join(tmpdir, f"{m}.nii.gz")
            with open(temp_path, "wb") as f:
                f.write(file_obj.read())

            # Cargar imagen desde archivo guardado
            img = nib.load(temp_path).get_fdata().astype(np.float32)

            # Aplicar N4 solo a t1n y t1c
            if m in ["t1n", "t1c"]:
                sitk_img = sitk.GetImageFromArray(img)
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations([50, 50, 50, 30])
                corrected = corrector.Execute(sitk_img)
                img = sitk.GetArrayFromImage(corrected)

            corrected_channels.append(img)

        volume = np.stack(corrected_channels, axis=-1)

        # Normalizar z-score
        for c in range(4):
            mean = volume[..., c].mean()
            std = volume[..., c].std()
            volume[..., c] = (volume[..., c] - mean) / std if std > 1e-5 else np.zeros_like(volume[..., c])

        # Resize a 128³
        zoom = (
            target_size / volume.shape[0],
            target_size / volume.shape[1],
            target_size / volume.shape[2]
        )
        resized = ndimage.zoom(volume, zoom + (1,), order=1)

        return resized.astype(np.float32)
# =========================
# Parámetros y utilidades
# =========================
PATCH_SIZE_64 = 64
PATCH_SIZE_96 = 96
NUM_CLASSES   = 5

MODALIDADES = ["t1n", "t1c", "t2f", "t2w"]
COLORES_RGB = {
    1: (1.0, 0.0, 0.0),  # NETC
    2: (0.0, 1.0, 0.0),  # SNFH
    3: (1.0, 1.0, 0.0),  # ET
    4: (0.0, 1.0, 1.0),  # RC
}

# flips TTA: (fz, fy, fx) con 0/1
FLIPS = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]

def minmax(img):
    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
    vmax = vmin + 1e-6 if vmax <= vmin else vmax
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

def overlay_segmentation(slice_image, slice_mask, alpha=0.5):
    img = minmax(slice_image)
    rgb = np.stack([img]*3, axis=-1)
    out = rgb.copy()
    for cls, color in COLORES_RGB.items():
        m = (slice_mask == cls)
        for c in range(3):
            out[..., c] = np.where(m, (1-alpha)*out[..., c] + alpha*color[c], out[..., c])
    return np.clip(out, 0, 1)

def volume_key(volume: np.ndarray, use_tta: bool, th_det: float, stride: int, model_sig: str) -> str:
    h = hashlib.md5()
    h.update(volume.tobytes())
    h.update(f"{use_tta}-{th_det}-{stride}-{model_sig}".encode())
    return h.hexdigest()

def flip3d(x, fz, fy, fx):
    """x: (D,H,W,C) o (D,H,W,classes). Devuelve vol volteado."""
    if fz: x = np.flip(x, axis=0)
    if fy: x = np.flip(x, axis=1)
    if fx: x = np.flip(x, axis=2)
    return x

# =========================
# Carga de modelos (cache)
# =========================
@st.cache_resource
def cargar_modelos():
    # Detectores (ensemble)
    det_auc    = load_model("/notebooks/best_detector_advanced.keras", compile=False)
    det_recall = load_model("/notebooks/best_detector_recall.keras",   compile=False)

    # Segmentadores
    seg_96 = load_model("/notebooks/runs/unetpp3d_ds_v2_96/best_unetpp3d_ds_96.keras", compile=False)
    seg_64 = load_model("/notebooks/runs/unetpp3d_ds_v2_20250814-223504/best_unetpp3d_ds.keras", compile=False)

    model_sig = "|".join([
        str(os.path.getmtime("/notebooks/best_detector_advanced.keras")),
        str(os.path.getmtime("/notebooks/best_detector_recall.keras")),
        str(os.path.getmtime("/notebooks/runs/unetpp3d_ds_v2_96/best_unetpp3d_ds_96.keras")),
        str(os.path.getmtime("/notebooks/runs/unetpp3d_ds_v2_20250814-223504/best_unetpp3d_ds.keras")),
    ])
    return (det_auc, det_recall), (seg_64, seg_96), model_sig

# =========================
# Detector y segmentadores
# =========================
def detectar_coords(detectors, volume, patch_size=PATCH_SIZE_64, stride=16, th=0.30):
    det_auc, det_rec = detectors
    D, H, W, _ = volume.shape
    coords = []
    for z in range(0, D - patch_size + 1, stride):
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size, :]
                patch = np.expand_dims(patch, 0).astype(np.float32)
                p1 = float(det_auc.predict(patch, verbose=0).squeeze())
                p2 = float(det_rec.predict(patch, verbose=0).squeeze())
                p  = 0.5*(p1+p2)
                if p >= th:
                    coords.append((z,y,x))
    return coords

def predict_seg_patch(model, patch):
    """Acepta salida con DS (lista/tuple) y quita batch; devuelve softmax (P,P,P,5)."""
    out = model.predict(patch, verbose=0)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.shape[0] == 1:
        out = out[0]
    return out.astype(np.float32)

def predict_with_tta(model, patch, patch_sz):
    """
    TTA con 8 flips; patch: (1,P,P,P,4); devuelve (P,P,P,5)
    """
    P = patch_sz
    acc = np.zeros((P,P,P,NUM_CLASSES), dtype=np.float32)
    for fz,fy,fx in FLIPS:
        p = patch[0]                                # (P,P,P,4)
        p = flip3d(p, fz, fy, fx)
        p = np.expand_dims(p, 0).astype(np.float32) # (1,P,P,P,4)
        pred = predict_seg_patch(model, p)          # (P,P,P,5)
        pred = flip3d(pred, fz, fy, fx)             # deshacer flip
        acc += pred
    return acc / float(len(FLIPS))

def ensemble_voxelwise_softmax(soft64, soft96_center):
    """Clases 1–3 del 96, clase 4 del 64, fondo promedio."""
    ens = np.zeros_like(soft64, dtype=np.float32)
    ens[..., 0] = 0.5*(soft64[..., 0] + soft96_center[..., 0])
    ens[..., 1] = soft96_center[..., 1]
    ens[..., 2] = soft96_center[..., 2]
    ens[..., 3] = soft96_center[..., 3]
    ens[..., 4] = soft64[..., 4]
    return ens

def inferir_segmentacion(detectors, segs, volume, use_tta=False, th_det=0.30, stride=16):
    """Devuelve (softmax_avg(128,128,128,5), pred_mask(128,128,128))."""
    (seg64, seg96) = segs
    D,H,W,C = volume.shape
    assert (D,H,W) == (128,128,128), "Se espera volumen 128³ preprocesado (D=H=W=128)."

    coords64 = detectar_coords(detectors, volume, patch_size=PATCH_SIZE_64, stride=stride, th=th_det)
    if len(coords64) == 0:
        # fallback denso
        coords64 = [(z,y,x)
                    for z in range(0, D - PATCH_SIZE_64 + 1, PATCH_SIZE_64)
                    for y in range(0, H - PATCH_SIZE_64 + 1, PATCH_SIZE_64)
                    for x in range(0, W - PATCH_SIZE_64 + 1, PATCH_SIZE_64)]

    softmax_volume = np.zeros((D,H,W,NUM_CLASSES), dtype=np.float32)
    count_volume   = np.zeros((D,H,W,NUM_CLASSES), dtype=np.float32)

    prog = st.progress(0.0)
    total = len(coords64)
    for i, (z,y,x) in enumerate(coords64, 1):
        # 64³
        p64 = volume[z:z+PATCH_SIZE_64, y:y+PATCH_SIZE_64, x:x+PATCH_SIZE_64, :]
        p64 = np.expand_dims(p64, 0).astype(np.float32)
        if use_tta:
            soft64 = predict_with_tta(seg64, p64, PATCH_SIZE_64)      # (64,64,64,5)
        else:
            soft64 = predict_seg_patch(seg64, p64)

        # 96³: centrado respecto al 64 (margen 16), con manejo de bordes
        z0 = np.clip(z - 16, 0, D - PATCH_SIZE_96)
        y0 = np.clip(y - 16, 0, H - PATCH_SIZE_96)
        x0 = np.clip(x - 16, 0, W - PATCH_SIZE_96)

        p96 = volume[z0:z0+PATCH_SIZE_96, y0:y0+PATCH_SIZE_96, x0:x0+PATCH_SIZE_96, :]
        p96 = np.expand_dims(p96, 0).astype(np.float32)

        # predicción (con o sin TTA)
        soft96 = predict_with_tta(seg96, p96, PATCH_SIZE_96) if use_tta else predict_seg_patch(seg96, p96)

        # en vez de usar [16:80], recorta usando el offset real del parche 64 dentro del 96
        off_z = z - z0
        off_y = y - y0
        off_x = x - x0
        soft96_c = soft96[off_z:off_z+PATCH_SIZE_64, off_y:off_y+PATCH_SIZE_64, off_x:off_x+PATCH_SIZE_64, :]

        # ensemble voxel-wise por clase
        ens = ensemble_voxelwise_softmax(soft64, soft96_c)            # (64,64,64,5)

        # acumular
        softmax_volume[z:z+PATCH_SIZE_64, y:y+PATCH_SIZE_64, x:x+PATCH_SIZE_64, :] += ens
        count_volume[z:z+PATCH_SIZE_64, y:y+PATCH_SIZE_64, x:x+PATCH_SIZE_64, :]   += 1.0

        prog.progress(i/total)

    count_volume[count_volume == 0] = 1.0
    softmax_avg = softmax_volume / count_volume
    pred_mask   = np.argmax(softmax_avg, axis=-1).astype(np.uint8)
    return softmax_avg, pred_mask

# =========================
# Inferencia cacheada
# =========================
@st.cache_data(show_spinner=False)
def inferir_cacheada(volume: np.ndarray, use_tta: bool, th_det: float, stride: int, model_sig: str):
    vol = np.array(volume, copy=True)  # asegurar hash estable
    detectors, segmentadores, _ = cargar_modelos()
    return inferir_segmentacion(detectors, segmentadores, vol, use_tta=use_tta, th_det=th_det, stride=stride)

# =========================
# UI
# =========================
st.set_page_config(page_title="BraTS-MET — Demo", layout="wide")
st.title("🧠 BraTS-MET — Detección, Segmentación y Volumetría")

# Bloque de subida doble
st.subheader("📂 Subida de datos")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Opción 1: subir volumen `.npy` ya preprocesado")
    uploaded_npy = st.file_uploader("Sube archivo `.npy` con shape (128,128,128,4)", type="npy", key="vol_npy")

with col2:
    st.markdown("### Opción 2: subir las 4 modalidades `.nii.gz`")
    modalidades_nii = {
        "t1n": st.file_uploader("T1n", type="nii.gz", key="t1n"),
        "t1c": st.file_uploader("T1c", type="nii.gz", key="t1c"),
        "t2f": st.file_uploader("T2-FLAIR", type="nii.gz", key="t2f"),
        "t2w": st.file_uploader("T2w", type="nii.gz", key="t2w")
    }

volume = None

# ✅ Si se subió un .npy directamente
if uploaded_npy is not None:
    volume = np.load(uploaded_npy)
    if volume.shape != (128,128,128,4):
        st.error("El volumen debe tener shape exacto (128,128,128,4).")
        st.stop()
    st.success("✅ Volumen `.npy` cargado correctamente.")

# ✅ Si se subieron las 4 modalidades
elif all(modalidades_nii.values()):
    if "volume_preprocessed" not in st.session_state:
        with st.spinner("📦 Preprocesando resonancias..."):
            volume = preprocess_uploaded_modalities(modalidades_nii)
            st.session_state["volume_preprocessed"] = volume
            st.success("✅ Modalidades `.nii.gz` preprocesadas correctamente.")
            np.save("volume_preprocessed.npy", volume)
            st.download_button(
                "⬇️ Descargar volumen preprocesado (.npy)",
                data=open("volume_preprocessed.npy", "rb").read(),
                file_name="volume_preprocessed.npy"
            )
    else:
        volume = st.session_state["volume_preprocessed"]
        st.success("✅ Modalidades preprocesadas previamente cargadas.")

st.sidebar.header("⚙️ Opciones de inferencia")
use_tta = st.sidebar.checkbox("🧪 TTA (8 flips)", value=False)
th_det  = st.sidebar.slider("Umbral detector", 0.0, 1.0, 0.30, 0.01)
stride  = st.sidebar.selectbox("Stride detector", [8 ,16, 32], index=0)

if volume is not None:
    (detectors, segmentadores, model_sig) = cargar_modelos()

    # clave única para cache/sesión
    key = volume_key(volume, use_tta, th_det, stride, model_sig)

    # botón de inferencia
    if st.button("🚀 Segmentar / Re-segmentar"):
        with st.spinner("Segmentando…"):
            softmax_avg, pred_mask = inferir_cacheada(volume, use_tta, th_det, stride, model_sig)
        st.session_state["seg_key"]   = key
        st.session_state["softmax"]   = softmax_avg
        st.session_state["pred_mask"] = pred_mask
        st.success("✅ Segmentación lista.")

    ready = ("pred_mask" in st.session_state) and (st.session_state.get("seg_key") == key)

    if ready:
        pred_mask  = st.session_state["pred_mask"]
        softmax_avg = st.session_state["softmax"]

        # Controles de visualización (no re-calculan inferencia)
        colA, colB, colC, colD = st.columns(4)
        with colA:
            canal = st.selectbox(" Modalidad", MODALIDADES, index=1)
        with colB:
            vista = st.selectbox(" Vista ", ["Axial", "Coronal", "Sagital"], index=0)
        with colC:
            slice_idx = st.slider(" Slice ", 0, 127, 64)
        with colD: 
            show_mask = st.checkbox(" Mostrar máscara", value=True)
            

        ch = MODALIDADES.index(canal)
        # Extraer el corte según la vista seleccionada
        if vista == "Axial":
            img_slice = volume[slice_idx, :, :, ch]
            msk_slice = pred_mask[slice_idx, :, :]
        elif vista == "Coronal":
            img_slice = volume[:, slice_idx, :, ch]
            msk_slice = pred_mask[:, slice_idx, :]
        else:  # Sagital
            img_slice = volume[:, :, slice_idx, ch]
            msk_slice = pred_mask[:, :, slice_idx]

        if show_mask:
            # Overlay con máscara
            overlay = overlay_segmentation(img_slice, msk_slice, alpha=0.55)

            # Figura con leyenda
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(overlay)
            ax.axis("off")
            legend = ax.legend(handles=legend_elements, loc="lower right", fontsize="6", frameon=True, framealpha = 0.8, handlelength=1.5, borderpad = 0.3, labelspacing = 0.2, borderaxespad = 0.3)
            legend.get_frame().set_boxstyle('Round', pad=0.2)
            st.pyplot(fig, use_container_width = False)

            # Botón para descargar imagen
            fig.savefig("overlay_segmented_with_legend.png", dpi=300, bbox_inches="tight")
            st.download_button("⬇️ Descargar overlay + leyenda (.png)",
                               data=open("overlay_segmented_with_legend.png", "rb").read(),
                               file_name="overlay_segmented_with_legend.png")

        else:
            # Solo la imagen original sin máscara
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(minmax(img_slice), cmap="gray")
            ax.axis("off")
            st.pyplot(fig, use_container_width = False)

        # Volumetría
        st.subheader("📐 Volumen por clase (voxeles ≈ mm³)")
        vols = {f"Clase {c}": int(np.sum(pred_mask == c)) for c in [1,2,3,4]}
        st.json(vols)

        # Descargas
        out_soft = "softmax_pred.npz"
        out_mask = "segmentation.npy"
        np.savez_compressed(out_soft, softmax=softmax_avg)
        np.save(out_mask, pred_mask)
        st.download_button("⬇️ Descargar softmax (.npz)", data=open(out_soft, "rb").read(), file_name=out_soft)
        st.download_button("⬇️ Descargar máscara (.npy)",  data=open(out_mask, "rb").read(), file_name=out_mask)

    else:
        st.info("Pulsa “Segmentar / Re-segmentar” para generar la predicción.")
else:
    st.info("Sube un volumen `.npy` o las 4 modalidades `.nii.gz` para empezar.")