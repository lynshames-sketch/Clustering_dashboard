# Suite R&D Core ML & Espace Latent (`Clustering_dashboard`)

> **Exploration de la Topologie Latente (*GMM* • *k-means* • *PyTorch*) & Réduction d'Hallucinations par Entropie**  
> **Auteur :** Saubaber Longang Gamo (Ph.D. en Économie • Modélisation Économétrique & IA)

[![License: MIT](https://img.shields.io/badge/License-MIT-58a6ff.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Palette Coolors](https://img.shields.io/badge/Coolors%20Palette-0d1117%20%7C%20161b22%20%7C%2058a6ff%20%7C%20bc8cff%20%7C%20f0f6fc-58a6ff)](https://coolors.co/0d1117-161b22-58a6ff-bc8cff-f0f6fc)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-58a6ff?logo=github)](https://lynshames-sketch.github.io/Clustering_dashboard/)

---

## 🎨 Charte Graphique & Palette Coolors AI Research

Palette officielle : 🔗 **[https://coolors.co/0d1117-161b22-58a6ff-bc8cff-f0f6fc](https://coolors.co/0d1117-161b22-58a6ff-bc8cff-f0f6fc)**

* `#0d1117` : Obsidian Dark / Fond de recherche sobre
* `#161b22` : Slate Card / Conteneurs & panneaux
* `#58a6ff` : Electric Ice Blue / Embeddings, vecteurs & attention
* `#bc8cff` : Quantum Lilac / Centroïdes, clusters & convergence
* `#f0f6fc` : Pure Cold White / Typographie scientifique haute netteté

---

## 📐 Spécifications Algorithmiques & Mathématiques ($\LaTeX$)

### 1. Perte Contrastive (*InfoNCE*)
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\sum_{j=1}^K \exp(\mathbf{q} \cdot \mathbf{k}_j / \tau)}$$

### 2. Coefficient de Silhouette
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$

### 3. Filtrage Anti-Hallucination par Entropie de Shannon
$$H(p) = -\sum_{c=1}^C p_c \log_2(p_c) \le H_{\text{seuil}}$$

---

## 🚀 Démarrage Rapide

```bash
git clone https://github.com/lynshames-sketch/Clustering_dashboard.git
cd Clustering_dashboard
python -m http.server 8086
```
Accédez ensuite à : **`http://localhost:8086/`**

---

## 👤 Auteur

**Saubaber Longang Gamo (Ph.D.)**  
- **LinkedIn :** [linkedin.com/in/saubaber-longang-18416216a](https://www.linkedin.com/in/saubaber-longang-18416216a)

---

## 📄 Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.
