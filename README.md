# Suite R&D Core ML & Espace Latent (`Clustering_dashboard`)

> **Exploration de la Topologie Latente (*GMM* • *k-means* • *PyTorch*) & Réduction d'Hallucinations par Entropie**  
> **Auteur :** Saubaber Longang Gamo (Ph.D. en Économie • Modélisation Économétrique & IA)

[![License: MIT](https://img.shields.io/badge/License-MIT-4f46e5.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-4f46e5?logo=github)](https://lynshames-sketch.github.io/Clustering_dashboard/)

---

## 🔬 Charte Graphique : Laboratoire Scientifique & White Paper

Une esthétique épurée inspirée des publications et laboratoires de recherche contemporains (Nature, OpenAI Research, DeepMind) :
* `#f8fafc` : **Blanc Perle / Papier Recherche** (Fond d'immersion lumineux avec micro-trame géométrique)
* `#ffffff` : **Cartes Blanches Porcelaine** (Conteneurs avec bordures délimitées fines `#e2e8f0`)
* `#0f172a` : **Ardoise Anthracite Profonde** (Titres et typographie haute lisibilité)
* `#4f46e5` : **Indigo Royal** (Embeddings, vecteurs d'attention et contrôles interactifs)
* `#7c3aed` : **Violet Quantique** (Métrique de Davies-Bouldin, centroïdes et courbes de convergence)
* `#059669` : **Émeraude Scientifique** (Inférences factuelles à faible entropie)
* `#dc2626` : **Rubis / Crimson** (Dérive sémantique et hallucinations rejetées)

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
