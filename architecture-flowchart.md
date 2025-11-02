# Architecture du Chatbot d'Analyse Structurelle

## Diagramme de Flux - Interface ChatGPT avec Routage Intelligent

```mermaid
flowchart TD
    A[Utilisateur accède à l'application] --> B[Interface ChatGPT style]
    B --> C{Fichier Excel<br/>déjà chargé?}
    
    C -->|Non| D[Affichage écran d'accueil<br/>Zone de téléchargement]
    C -->|Oui| E[Interface chat active<br/>avec historique restauré]
    
    D --> F[Utilisateur télécharge<br/>fichier Excel]
    F --> G[Backend FastAPI traite le fichier]
    G --> H[Extraction données<br/>Colonnes 0-1 et 4-5]
    H --> I[Calcul métriques d'ingénierie<br/>11 types de graphiques]
    I --> J[Stockage en session navigateur]
    J --> K[Message de confirmation<br/>+ Boutons Change File & Clear Chat]
    K --> E
    
    E --> L[Utilisateur saisit une question]
    L --> M[Système de routage intelligent]
    
    M --> N{Contient mots-clés<br/>visuels?<br/>plot/show/display}
    
    N -->|OUI| O[Détection type de graphique demandé]
    O --> P{Quel type?}
    
    P -->|envelope/backbone/unified| Q[Courbes enveloppe unifiées]
    P -->|hysteresis/force-displacement| R[Courbes hystérésis]
    P -->|energy/cumulative/dissipation| S[Dissipation énergie cumulative]
    P -->|force history| T[Historique force vs temps]
    P -->|loading history| U[Historique chargement]
    P -->|comparison/compare| V[Graphique comparatif]
    P -->|all/graphs/plots| W[Tous les 11 graphiques]
    P -->|ductility explanation| X[Explication calcul ductilité]
    P -->|bilinear| Y[Idéalisation bilinéaire]
    
    Q --> Z[Génération graphique Plotly]
    R --> Z
    S --> Z
    T --> Z
    U --> Z
    V --> Z
    X --> Z
    Y --> Z
    W --> AA[Génération multiple<br/>Un graphique par message]
    AA --> BB[Affichage séquentiel vertical]
    
    Z --> CC[Affichage graphique interactif<br/>dans le chat]
    BB --> CC
    
    N -->|NON| DD{Question sur<br/>données simples?<br/>max/min/ductility}
    
    DD -->|OUI| EE[Réponse directe avec données<br/>Sans appel IA]
    EE --> FF[Affichage valeurs numériques<br/>formatées]
    
    DD -->|NON| GG[Appel API GPT-4o<br/>pour analyse complexe]
    GG --> HH[Génération réponse technique<br/>avec contexte données]
    HH --> II[Affichage réponse formatée<br/>Markdown]
    
    CC --> JJ[Ajout message à l'historique]
    FF --> JJ
    II --> JJ
    JJ --> KK[Sauvegarde session navigateur]
    KK --> LL[Attente nouvelle question]
    LL --> L
    
    E --> MM[Bouton Clear Chat cliqué?]
    MM -->|OUI| NN[Confirmation utilisateur]
    NN -->|Confirmé| OO[Effacement historique chat<br/>Conservation données fichier]
    OO --> PP[Message bienvenue avec résumé]
    PP --> E
    NN -->|Annulé| E
    MM -->|NON| L
    
    E --> QQ[Bouton Change File cliqué?]
    QQ -->|OUI| F
    QQ -->|NON| L
    
    subgraph "Types de Graphiques Disponibles"
        RR[1. Référence: Courbe Force-Déplacement]
        SS[2. Test: Courbe Force-Déplacement] 
        TT[3. Comparaison: Référence vs Test]
        UU[4. Historique Chargement]
        VV[5. Historique Force]
        WW[6. Dissipation Énergie Cumulative]
        XX[7. Enveloppe Référence]
        YY[8. Enveloppe Test]
        ZZ[9. Comparaison Enveloppes]
        AAA[10. Explication Ductilité]
        BBB[11. Idéalisation Bilinéaire]
    end
    
    subgraph "Données Session"
        CCC[Historique conversation]
        DDD[Données Excel traitées]
        EEE[Statistiques calculées]
        FFF[Métriques d'ingénierie]
    end
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style M fill:#fff3e0
    style O fill:#e8f5e8
    style GG fill:#fff8e1
    style Z fill:#fce4ec
    style JJ fill:#f1f8e9
```

## Légende des Couleurs

- 🔵 **Bleu clair** - Points d'entrée utilisateur
- 🟣 **Violet clair** - Interface principale
- 🟠 **Orange clair** - Système de routage
- 🟢 **Vert clair** - Génération graphiques
- 🟡 **Jaune clair** - Traitement IA
- 🌸 **Rose clair** - Affichage résultats
- 🍃 **Vert tendre** - Gestion session

## Stratégie de Routage Intelligent

### 1. Priorisation Visuelle (PRIORITÉ 1)
- **Mots-clés détectés**: `plot`, `show`, `display`, `visualize`, `graph`, `chart`
- **Action**: Génération immédiate de graphiques Plotly interactifs
- **Résultat**: Graphiques affichés inline dans le chat

### 2. Réponses Données Directes (PRIORITÉ 2)
- **Types de questions**: 
  - "What's the max force?"
  - "Show me the ductility"
  - "What's the stiffness ratio?"
- **Action**: Calcul direct sans IA
- **Résultat**: Valeurs numériques formatées instantanément

### 3. Analyse IA Complexe (PRIORITÉ 3)
- **Modèle utilisé**: GPT-4o
- **Types de questions**:
  - Interprétations techniques
  - Recommandations ingénierie
  - Analyses comparatives approfondies
- **Action**: Traitement contextualisé avec toutes les données

## Types de Graphiques Supportés

| Type | Description | Mots-clés de Détection |
|------|-------------|------------------------|
| 1. Hystérésis Référence | Courbe Force-Déplacement cas de référence | `reference`, `hysteresis` |
| 2. Hystérésis Test | Courbe Force-Déplacement données test | `test`, `BCJS`, `specimen` |
| 3. Comparaison | Référence vs Test superposés | `comparison`, `compare`, `vs` |
| 4. Historique Chargement | Déplacement vs Temps | `loading history`, `displacement history` |
| 5. Historique Force | Force vs Temps | `force history` |
| 6. Énergie Cumulative | Dissipation d'énergie | `energy`, `cumulative`, `dissipation` |
| 7. Enveloppe Référence | Courbe enveloppe référence | `envelope`, `reference` |
| 8. Enveloppe Test | Courbe enveloppe test | `envelope`, `test` |
| 9. Comparaison Enveloppes | Enveloppes superposées | `envelope comparison` |
| 10. Explication Ductilité | Calcul ductilité annoté | `ductility explanation` |
| 11. Idéalisation Bilinéaire | Modèle bilinéaire | `bilinear`, `idealization` |

## Fonctionnalités Interface

### Boutons de Contrôle
- **Upload File**: Visible au démarrage uniquement
- **Change File**: Visible après chargement, permet de changer le fichier
- **Clear Chat**: Visible après chargement, efface l'historique mais garde les données

### Gestion Session Navigateur
- **Données conservées**: 
  - Fichier Excel traité
  - Statistiques calculées
  - Métriques d'ingénierie
  - Historique conversation
- **Restauration**: Rechargement page = interface complètement restaurée
- **Effacement**: Fermeture onglet = perte de toutes les données

## Architecture Technique

### Backend FastAPI
- **Endpoint principal**: `/chat` - Routage intelligent des requêtes
- **Endpoint upload**: `/upload` - Traitement fichiers Excel
- **Moteur graphique**: Plotly pour visualisations interactives
- **IA**: OpenAI GPT-4o pour analyses complexes

### Frontend
- **Style**: Interface ChatGPT (messages bulles, centré, défilement)
- **Graphiques**: Intégration Plotly.js inline dans les messages
- **Stockage**: Session Storage navigateur pour persistance
- **Responsive**: Adaptatif mobile/desktop

## Format de Données Excel

### Structure Attendue
- **Colonnes 0-1**: Cas de référence (U mm, F kN)
- **Colonnes 4-5**: Données test (u mm, RF kN)
- **Ligne 1**: En-têtes
- **Ligne 2+**: Données numériques
- **Cellule [0,4]**: Nom du spécimen test

### Métriques Calculées Automatiquement
- Ductilité de déplacement
- Rigidité initiale
- Dissipation d'énergie totale
- Facteur de comportement (q)
- Classification de ductilité
- Ratios comparatifs (%)

---

*Architecture générée pour l'Assistant d'Analyse Structurelle - Interface ChatGPT avec Routage Intelligent*