# 🧠 BraTS-METS 2025 — Detección y Segmentación 3D de Metástasis Cerebrales

Demo académica (no uso clínico) de un pipeline 3D en **cascada**: **Detector binario → Segmentadores UNet++** con **TTA** y **ensemble por clase**. Incluye **app Streamlit** para probar con volúmenes propios.

---

## 🎯 Objetivo

- **Detectar** parches candidatos con alta **sensibilidad (recall)**.
- **Segmentar** voxel-wise en 5 clases (fondo + 4 etiquetas BraTS-METS).
- **Productivizar** el flujo en una **app web** con overlays y volumetría.

**Etiquetas BraTS-METS**
| ID | Clase | Descripción breve |
|---:|:------|:------------------|
| 1 | **NETC** | Núcleo tumoral no realzante (dentro de ET). |
| 2 | **SNFH** | Hiperseñal FLAIR perilesional (edema/infiltración). |
| 3 | **ET** | Tumor realzante en T1 post-contraste. |
| 4 | **RC** | Cavidad de resección (post-quirúrgica). |

---

## 🗂️ Contenidos del repo

- `TFM_<nombre>.ipynb` — Notebook principal (pipeline completo).
- `Notebook_HTML/TFM_notebook.html` — Notebook exportado a HTML con anclas por sección.
- `app.py` — App **Streamlit** (detección + segmentación + visualización).
- `anexos/` — Snippets de código utilizados en la memoria (Word).
- `assets/` — Figuras (resultados con/sin TTA, capturas de la app, etc.).
- `requirements.txt` — Dependencias sugeridas.

> **Datos y modelos** no incluidos por tamaño.

---

## 🧩 Metodología (resumen)

**Preprocesado**
- Unificación a **128×128×128×4** (T1n, T1c, FLAIR, T2).
- **N4** (T1n/T1c), **z-score por canal** (solo tejido), **resize**: trilineal (imagen) / vecino más cercano (máscara).

**Parches**
- Dos tamaños: **64³** (detalle/lesiones pequeñas) y **96³** (más contexto).
- Oversampling de clases minoritarias + parches negativos.

**Detector binario (ensemble)**
- Dos detectores 3D combinados por media (**AUC + Recall**).
- **Umbral operativo `t = 0.30`** (elegido por **PR/F2**) ⇒ prioriza **recall**.
- **Stride `16`** con parche `64³` (≈75% de solape) para no perder lesiones en bordes.

**Segmentadores**
- **UNet++ 3D con Deep Supervision** en **64³** y **96³**.
- **TTA** (flips 3D) + **ensemble voxel-wise**:
  - **Clases 1–3** ⇢ modelo **96³**  
  - **Clase 4 (RC)** ⇢ modelo **64³**  
  - **Fondo** ⇢ promedio de ambos

**Métricas**
- **Dice** (solape), **NSD** (acuerdo de superficies con tolerancia), **HD95**.

---

## 🚀 Cómo ejecutar

### 1) Entorno
```bash
# conda (recomendado)
conda create -n bratsmets python=3.10 -y
conda activate bratsmets
pip install -r requirements.txt
