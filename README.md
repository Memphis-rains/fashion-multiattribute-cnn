# Fashion Multi-Attribute CNN

A lightweight end-to-end pipeline that automatically tags **fashion product images** with six key attributes:

- **Gender** (Men, Women, Boys, Girls, Unisex)  
- **Master Category** (Apparel, Footwear, Accessories, etc.)  
- **Sub-Category** (Topwear, Bottomwear, Bags, …)  
- **Article Type** (100 + fine-grained classes)  
- **Base Colour** (≈ 45 colours)  
- **Usage** (Casual, Formal, Sports, …)

The model employs a shared convolutional backbone with six soft-max heads, allowing a single forward pass to predict all attributes.

---

## Dataset  

| Source | Kaggle – Fashion Product Images (Small) |
|--------|-----------------------------------------|
| Link   | <https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small> |
| Size   | ≈ 25 000 JPG images (80 × 60 px) + `styles.csv` annotations |

---

## Quick Metrics  

| Attribute        | Validation Accuracy |
|------------------|---------------------|
| MasterCategory   | **94 %** |
| Usage            | 86 % |
| Gender           | 83 % |
| SubCategory      | 84 % |
| BaseColour       | 58 % |
| ArticleType      | < 1 % |
| **Overall (micro-avg)** | **78 %** |

*NB: fine-grained **ArticleType** suffers from class imbalance; future work will focus on boosting this score.*

