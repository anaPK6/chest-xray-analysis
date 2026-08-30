"""Curated radiology knowledge base for the RAG chatbot.

Small, controlled corpus describing the 5 CheXpert pathologies the model
predicts. Each (pathology, section) pair becomes one retrievable chunk. Kept as
plain Python so there's no data file to ship or license.

Sources are educational/reference-level summaries — NOT clinical guidance.
"""
from __future__ import annotations

# pathology -> {section: text}
KB: dict[str, dict[str, str]] = {
    "Atelectasis": {
        "Definition": (
            "Atelectasis is the partial or complete collapse of a lung or a lobe, "
            "where the alveoli deflate and lose air volume. It is one of the most "
            "common findings on chest radiographs."
        ),
        "Radiographic signs": (
            "On a chest X-ray, atelectasis appears as increased opacity in the "
            "affected region with volume loss. Signs include displacement of fissures, "
            "crowding of vessels and bronchi, elevation of the hemidiaphragm, and "
            "shift of the mediastinum or trachea toward the collapsed area."
        ),
        "Causes": (
            "Common causes include airway obstruction (mucus plugging, tumor, foreign "
            "body), compression from pleural effusion or pneumothorax, and post-operative "
            "hypoventilation. Shallow breathing after surgery is a frequent cause."
        ),
        "Clinical significance": (
            "Small atelectasis may be asymptomatic; larger collapse can cause shortness of "
            "breath and reduced oxygenation. It can also mask or mimic other pathology, so "
            "it is often noted alongside effusions or consolidation."
        ),
    },
    "Cardiomegaly": {
        "Definition": (
            "Cardiomegaly means an enlarged heart. On imaging it is commonly assessed by "
            "the cardiothoracic ratio — the width of the heart relative to the chest."
        ),
        "Radiographic signs": (
            "On a frontal chest X-ray, cardiomegaly is suggested when the cardiothoracic "
            "ratio exceeds roughly 0.5 (heart width more than half the chest width). The "
            "cardiac silhouette appears enlarged; specific chamber enlargement can alter "
            "the contour of the heart border."
        ),
        "Causes": (
            "Causes include hypertension, coronary artery disease, cardiomyopathy, valvular "
            "heart disease, and pericardial effusion (which enlarges the silhouette without "
            "true muscle enlargement). It can reflect chronic strain on the heart."
        ),
        "Clinical significance": (
            "Cardiomegaly is a marker of underlying cardiac disease and may accompany heart "
            "failure. On X-ray it is often seen together with signs of pulmonary edema or "
            "pleural effusion when heart failure is present."
        ),
    },
    "Consolidation": {
        "Definition": (
            "Consolidation occurs when the air-filled spaces of the lung (alveoli) fill with "
            "fluid, pus, blood, or cells, making the lung tissue solid rather than aerated."
        ),
        "Radiographic signs": (
            "Consolidation appears as a region of increased opacity, often with air "
            "bronchograms (dark branching airways visible against the white consolidated "
            "lung). Unlike atelectasis, consolidation typically does not cause volume loss."
        ),
        "Causes": (
            "The classic cause is pneumonia (infection filling alveoli with pus). Other "
            "causes include pulmonary edema, pulmonary hemorrhage, aspiration, and less "
            "commonly malignancy or inflammatory disease."
        ),
        "Clinical significance": (
            "Consolidation usually indicates an active process such as infection and warrants "
            "clinical correlation. Its distribution (lobar vs patchy) helps narrow the cause."
        ),
    },
    "Edema": {
        "Definition": (
            "Pulmonary edema is the accumulation of excess fluid in the lung's interstitial "
            "spaces and alveoli, most often due to raised pressure in the pulmonary vessels."
        ),
        "Radiographic signs": (
            "Signs include bilateral perihilar haziness (a 'bat-wing' pattern), Kerley B "
            "lines (short horizontal lines at the lung periphery), peribronchial cuffing, "
            "and often an enlarged heart and pleural effusions when cardiac in origin."
        ),
        "Causes": (
            "The most common cause is cardiogenic — left heart failure raising pulmonary "
            "venous pressure. Non-cardiogenic causes include ARDS, kidney failure with fluid "
            "overload, and high-altitude exposure."
        ),
        "Clinical significance": (
            "Pulmonary edema causes breathlessness and low oxygen and can be life-threatening. "
            "On chest X-ray it frequently co-occurs with cardiomegaly and pleural effusion, "
            "which is why these labels often rise together."
        ),
    },
    "Pleural Effusion": {
        "Definition": (
            "A pleural effusion is a build-up of fluid in the pleural space — the thin gap "
            "between the lung and the chest wall."
        ),
        "Radiographic signs": (
            "On an upright chest X-ray, effusion causes blunting of the costophrenic angle "
            "(loss of the sharp corner where diaphragm meets ribs) and a meniscus-shaped "
            "fluid level. Large effusions can opacify much of a hemithorax and push the "
            "mediastinum away."
        ),
        "Causes": (
            "Causes include heart failure (the commonest), infection (parapneumonic effusion, "
            "empyema), malignancy, pulmonary embolism, and low blood protein states. Fluid is "
            "classified as transudate or exudate based on its composition."
        ),
        "Clinical significance": (
            "Effusions can compress the lung and cause breathlessness. The underlying cause "
            "matters clinically; on imaging effusion is frequently associated with edema, "
            "cardiomegaly, or adjacent consolidation."
        ),
    },
}


def kb_chunks() -> list[dict]:
    """Flatten the KB into retrievable chunks: {id, pathology, section, text}."""
    chunks = []
    for pathology, sections in KB.items():
        for section, text in sections.items():
            chunks.append({
                "id": f"{pathology}::{section}",
                "pathology": pathology,
                "section": section,
                # embed the label with the text so retrieval keys on both
                "text": f"{pathology} — {section}: {text}",
            })
    return chunks
