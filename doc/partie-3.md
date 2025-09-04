#  Partie 3 – Enjeux sociétaux et éthiques

---

## Identification des biais

### Biais critiques observés
- **Temporel (8/10)** : Données limitées à ~2 jours → pas de saisonnalité, modèle peu généralisable  
- **Montant (7/10)** : Fraudes concentrées sur faibles montants → sous-détection gros montants  
- **Géographique (6/10)** : Données uniquement européennes → inadaptées à d’autres contextes  

Risque moyen global : 7/10 

---

## Visualisations clés

### Patterns horaires
- Transactions normales : pic entre **8h et 20h**  
- Transactions frauduleuses : présentes jour & nuit, surreprésentées la nuit  
- **Insight** : ≈ **x3 plus de fraudes la nuit** que le jour  

### Distribution des montants
- Fraudes → médiane ≈ **9€**  
- Normales → médiane ≈ **23€**  
- **Insight** : les fraudes ciblent souvent de **petits montants**  

---

## Plan d’atténuation prioritaire

### Actions top 3
1. **Diversification temporelle** → collecter ≥12 mois de données (P1 – 3 mois)  
2. **Rééquilibrage montants** → génération synthétique (SMOTE) pour gros montants (P1 – 2 semaines)  
3. **Monitoring dérive** → alertes automatiques de performance en temps réel (P1 – 1 semaine)  

---

## Charte éthique FLUZZ – IA fraude

### Mission
> Protéger les clients contre la fraude avec équité et transparence.

### Principes
- **ÉQUITÉ** : égalité de traitement entre tous les profils  
- **TRANSPARENCE** : explication des décisions automatisées  
- **RESPONSABILITÉ** : supervision humaine obligatoire  

### Protection des données
- Anonymisation (PCA)  
- Chiffrement + accès contrôlé  
- Conservation limitée dans le temps  

### Droits clients
- Information claire sur usage IA  
- Recours contre décisions automatisées  
- Explication des blocages  

### Engagements mesurables (KPIs)
| Métrique          | Objectif  | Fréquence   |
|-------------------|-----------|-------------|
| **Fairness Score** | >0.85     | Mensuel     |
| **Faux Positifs**  | <3%       | Temps réel  |
| **Temps Recours**  | <24h      | Temps réel  |

---

## Conformité RGPD et AI Act

### Risque de ré-identification
- ~**85% unicité** des couples `(Time, Amount)` → **Risque élevé**  

### Actions correctives
- Ajout de bruit différentiel sur montants  
- Discrétisation temporelle (créneaux 4h)  
- Interface client pour rectification des données  

### Alignement réglementaire
- **RGPD** : protection & droits utilisateurs respectés  
- **AI Act** : système classé *Haut Risque* (finance)  
- **DPO** : validation des traitements
