import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Configuration ---
MODEL_PATH = r'C:\Users\patel\OneDrive\Desktop\FPR\models4\best_model_bs16.h5'  # path to your saved .h5 model
IMAGE_SIZE = (128, 128)

# Replace with your actual label mappings
# Keys must match model.output_names
label_encoders = {
    'gender':   ['Men', 'Women', 'Boys', 'Girls', 'Unisex'],
    'masterCategory': [
        'Apparel','Accessories','Footwear','Personal Care','Free Items',
        'Sporting Goods','Home'
    ],
    'subCategory': [
        'Bottomwear','Topwear','Socks','Watches','Shoes','Belts','Flip Flops',
        'Bags','Innerwear','Shoe Accessories','Jewellery','Lips','Saree',
        'Fragrance','Sandal','Nails','Dress','Wallets','Eyewear','Headwear',
        'Loungewear and Nightwear','Skin Care','Free Gifts','Mufflers','Ties',
        'Makeup','Accessories','Scarves','Beauty Accessories','Water Bottle',
        'Apparel Set','Skin','Eyes','Sports Accessories','Cufflinks',
        'Sports Equipment','Stoles','Gloves','Hair','Perfumes',
        'Home Furnishing','Bath and Body','Umbrellas','Wristbands','Vouchers'
    ],
    'articleType': [
        'Track Pants','Tshirts','Shirts','Socks','Watches','Casual Shoes','Belts',
        'Flip Flops','Handbags','Tops','Bra','Shoe Accessories','Sweatshirts',
        'Formal Shoes','Bracelet','Lipstick','Flats','Kurtas','Waistcoat',
        'Sports Shoes','Shorts','Briefs','Sarees','Perfume and Body Mist','Heels',
        'Sandals','Pendant','Nail Polish','Laptop Bag','Rain Jacket','Dresses',
        'Skirts','Wallets','Blazers','Ring','Sunglasses','Clutches','Shrug','Caps',
        'Innerwear Vests','Earrings','Trousers','Boxers','Backpacks','Jeans',
        'Jewellery Set','Dupatta','Capris','Bath Robe','Tunics','Jackets',
        'Lounge Pants','Face Wash and Cleanser','Necklace and Chains','Duffel Bag',
        'Sports Sandals','Deodorant','Sweaters','Free Gifts','Tracksuits',
        'Shoe Laces','Trunk','Mufflers','Bangle','Ties','Highlighter and Blush',
        'Travel Accessory','Lip Care','Scarves','Messenger Bag','Compact',
        'Night suits','Leggings','Eye Cream','Lip Gloss','Kurtis','Accessory Gift Set',
        'Beauty Accessory','Jumpsuit','Nightdress','Water Bottle','Kurta Sets',
        'Face Moisturisers','Suspenders','Mobile Pouch','Lip Liner','Robe',
        'Stockings','Kajal and Eyeliner','Eyeshadow','Headband','Patiala',
        'Camisoles','Tights','Lounge Tshirts','Fragrance Gift Set',
        'Face Scrub and Exfoliator','Lounge Shorts','Baby Dolls','Foundation and Primer',
        'Wristbands','Salwar and Dupatta','Swimwear','Tablet Sleeve',
        'Ties and Cufflinks','Footballs','Stoles','Shapewear','Salwar','Cufflinks',
        'Concealer','Rompers','Jeggings','Gloves','Sunscreen','Booties',
        'Mask and Peel','Waist Pouch','Rucksacks','Basketballs','Hair Colour',
        'Churidar','Clothing Set','Mascara','Nehru Jackets','Cushion Covers',
        'Key chain','Toner','Lip Plumper','Nail Essentials','Umbrellas',
        'Face Serum and Gel','Body Lotion','Makeup Remover','Rain Trousers',
        'Hat','Trolley Bag','Lehenga Choli','Ipad'
    ],
    'baseColour': [
        'Black','Grey','Green','Navy Blue','Blue','White','Beige','Bronze','Brown',
        'Copper','Pink','Off White','Maroon','Purple','Khaki','Silver','Orange',
        'Coffee Brown','Red','Charcoal','Gold','Steel','Multi','Magenta',
        'Sea Green','Cream','Yellow','Olive','Grey Melange','Teal','Burgundy',
        'Peach','Rust','Skin','Turquoise Blue','Tan','Lavender','Metallic',
        'Mustard','Mauve','Nude','Rose','Mushroom Brown','Taupe','Lime Green',
        'Fluorescent Green'
    ],
    'usage': ['Casual','Ethnic','Sports','Formal','Smart Casual','Party','Travel','Home']
}

# Load the model
@st.cache_resource
def load_cnn_model(path):
    return load_model(path)

model = load_cnn_model(MODEL_PATH)

# --- Streamlit Layout ---
st.title("Fashion Attribute Classification")
st.write("Upload an image to classify its fashion attributes.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=['jpg', 'png', 'jpeg']
)

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_batch)
    
    st.subheader("Predicted Attributes:")
    confidences = []

    # Ensure preds is list
    if not isinstance(preds, list):
        preds = [preds]

    for i, probs in enumerate(preds):
        # get output name
        attr_name = model.output_names[i]
        # fallback safe labels
        labels = label_encoders.get(attr_name, [str(j) for j in range(len(probs[0]))])
        # prediction index and confidence
        idx = int(np.argmax(probs[0]))
        label = labels[idx]
        confidence = float(probs[0][idx])
        # display
        st.write(f"**{attr_name}:** {label} ({confidence:.2%})")
        confidences.append(confidence)

    # Overall average confidence
    avg_conf = np.mean(confidences)
    st.markdown(f"---\n**Overall Confidence:** {avg_conf:.2%}")

    # Optional detailed probabilities
    if st.checkbox("Show detailed probabilities"):
        for i, probs in enumerate(preds):
            attr_name = model.output_names[i]
            labels = label_encoders.get(attr_name, [str(j) for j in range(len(probs[0]))])
            st.write(f"### {attr_name} Probabilities")
            for j, p in enumerate(probs[0]):
                st.write(f"- {labels[j]}: {p:.2%}")
