# Core ML & Latent Space Clustering Suite (`Clustering_dashboard`)

> **Exploration de l'Espace Latent, Visualiseur de Tenseurs & Réduction d'Hallucinations**  
> **Auteur :** Saubaber Longang Gamo (Ph.D. en Économie • Modélisation Économétrique & IA)

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-06b6d4?logo=github)](https://lynshames-sketch.github.io/Clustering_dashboard/)

---

## 🔬 Présentation de l'Architecture (R&D / Core ML)

Ce dashboard s'inscrit dans l'axe **3. Recherche & Core ML (R&D)** destiné aux laboratoires de recherche, universités et postes de Senior Data Scientist.

Il met en œuvre le tandem technologique :
* **Google Stitch (Vibe Design & Interface Réactive)** : Une interface transparente et épurée en *Glassmorphism* (Tailwind CSS, Canvas HTML5) affichant les cartes d'attention Transformer, la décomposition des tenseurs d'activation et la topologie des clusters.
* **Google Antigravity (Moteur d'Agents & Inférence)** : Orchestration locale des calculs vectoriels, calcul en temps réel des métriques de géométrie différentielle (Score de Silhouette, Davies-Bouldin) et filtrage des dérives sémantiques (hallucinations).

---

## 📐 Fondements Mathématiques & Métriques

### 1. Modélisation de l'Espace Latent ($z \in \mathbb{R}^d$)
- Partitionnement de mélanges gaussiens (GMM) avec estimation de densité :
  $$p(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k)$$
- **Score de Silhouette ($s \in [-1, 1]$)** : Évaluation de la cohésion intra-cluster par rapport à la séparation inter-clusters :
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- **Indice de Davies-Bouldin ($DB$)** : Mesure de similarité moyenne entre chaque cluster et son plus proche voisin.

### 2. Détection & Filtrage des Hallucinations Sémantiques
- Calcul de l'entropie prédictive token par token :
  $$H(X) = -\sum_{i=1}^V p(x_i) \log p(x_i)$$
- Les représentations situées au-delà du seuil critique d'entropie ($H > \theta$) sont isolées comme dérives hors-distribution et filtrées hors du sous-espace factuel (*manifold*).

### 3. Visualisation de Tenseurs & Matrice d'Attention
- Inspection interactive des tenseurs PyTorch : `[Batch, Seq_Len, D_model]`
- Visualisation de la matrice de produit scalaire adouci (*Scaled Dot-Product Attention*) :
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## 🚀 Démarrage Rapide

### 1. Cloner le Dépôt
```bash
git clone https://github.com/lynshames-sketch/Clustering_dashboard.git
cd Clustering_dashboard
```

### 2. Démarrer le Serveur Local
```bash
python -m http.server 8086
```
Puis accédez à : **`http://localhost:8086/`**

### 3. Exécuter le Script d'Évaluation Vectorielle
```bash
python clustering_core.py
```

---

## 👤 Auteur

**Saubaber Longang Gamo (Ph.D.)**  
- **LinkedIn :** [linkedin.com/in/saubaber-longang-18416216a](https://www.linkedin.com/in/saubaber-longang-18416216a)

---

## 📄 Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.
