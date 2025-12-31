from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Kullanılan veri seti: "phone_price_dataset.csv"

# Veri setindeki veriler;
# Name: Telefonun tam adı
# Brand: Telefonun markası
# Model: Telefonun modeli
# Battery capacity (mAh): Pil kapasitesi
# Screen size (inches): Ekran boyutu
# Touchscreen: Dokunmatik ekrana sahip olma
# Resolution x: Ekranın Xaxis'te çözünürlüğü
# Resolution y: Ekranın Yaxis'te çözünürlüğü
# Processor: İşlemci
# RAM (MB): RAM miktarı
# Internal storage (GB): Depolama
# Rear camera: Arka kameranın megapiksel(MP) bazında çözünürlüğü
# Front camera: Arka kameranın megapiksel(MP) bazında çözünürlüğü
# Operating system: Telefonun işletim sistemi
# Wi-Fi: Wi-Fi özelliği bulundurma
# Bluetooth: Bluetooth özelliği bulundurma
# GPS: GPS özelliği bulundurma
# Number of SIMs: Telefonda bulunan SIM yuvası sayısı
# 3G: 3G özelliği bulundurma
# 4G/ LTE: 4G/ LTE özelliği bulundurma
# Price(INR): Telefonun Hint Rupisi bazında fiyatı


pd.set_option('display.max_columns', 50)


phone_dataset = pd.read_csv("phone_price_dataset.csv").drop(columns=["Name","Model","Operating system","Touchscreen"]) # CSV dosyasından Pandas Dataframe oluşturdum. Kullanmak istemediğim sütunları kaldırdım.
phone_dataset = phone_dataset.rename(columns={ # Kullanım/Anlaşılma kolaylığı için özelliklerin isimlerini değiştirdim.
    "Screen size (inches)" : "Screen_Size",
    "Battery capacity (mAh)": "Battery_Capacity",
    "Resolution x": "Resolution_X",
    "Resolution y": "Resolution_Y",
    "Internal storage (GB)": "Internal_Storage_GB",
    "Rear camera": "Rear_Camera_MP",
    "Front camera": "Front_Camera_MP",
    "Number of SIMs": "SIM_Count",
    "4G/ LTE": "4G"
})

phone_dataset[["Bluetooth","Wi-Fi","GPS","3G","4G"]] = phone_dataset[["Bluetooth","Wi-Fi","GPS","3G","4G"]].replace({"Yes": 1, "No": 0})

phone_dataset_encoded = pd.get_dummies(
    phone_dataset,
    columns=["Brand"],
    drop_first=True
)# Markaların fiyatı etkilemesi faktörünün de hesaba katılabilmesi için ChatGPT yardımıyla One-Hot Encoding öğrenip kullandım.

LR = LinearRegression()

X = phone_dataset_encoded.drop(columns=["Price(INR)"])
y = phone_dataset_encoded["Price(INR)"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

LR.fit(X_train, y_train)

y_prediction = LR.predict(X_test)

karsilastirma_tablo = pd.DataFrame({ # Tahmin ve gerçek fiyatları kolayca karşılaştırmak için Pandas Dataframe kullandım
    'Gerçek Fiyat': y_test,
    'Tahmini Fiyat': y_prediction
})

mae = mean_absolute_error(y_test, y_prediction)

print(karsilastirma_tablo)
print(f"Ortalama Mutlak Hata: {mae:.2f}")

