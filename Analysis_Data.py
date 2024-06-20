import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Housing Price Dataset/train.csv/train.csv")
# Thay thế các giá trị 9 bằng NaN
df.replace(9, np.nan, inplace=True)
# Xóa cột id
df = df.drop(columns=['id'])
# Kiểm tra số lượng giá trị missing trong cột 'No. of Bedrooms' và các cột tiện nghi có nhiều giá trị khác nhau
missing_bedrooms = df['No. of Bedrooms'].isnull().sum()
# print(f'Số lượng giá trị thiếu trong cột "No. of Bedrooms": {missing_bedrooms}')

df['No. of Bedrooms'] = df['No. of Bedrooms'].fillna(
    df['No. of Bedrooms'].mean())
missing_bedrooms_after = df['No. of Bedrooms'].isnull().sum()
# print(f'Số lượng giá trị thiếu trong cột "No. of Bedrooms" sau khi chỉnh sửa: {missing_bedrooms_after}')


amenity_columns = ['MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack', 'RainWaterHarvesting',
                   'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security',
                   'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
                   'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave',
                   'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']
missing_amenity = df[amenity_columns].isnull().sum()
# print(f'Số lượng giá trị thiếu trong cột "Amenity": {missing_amenity}')


# Điền giá trị 0 cho các cột tiện nghi có nhiều giá trị thiếu
df[amenity_columns] = df[amenity_columns].fillna(0)

missing_amenity_after = df[amenity_columns].isnull().sum()

# Biểu đồ phân bố giá nhà
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Biểu đồ phân bố diện tích
plt.figure(figsize=(10, 6))
sns.histplot(df['Area'], bins=50, kde=True)
plt.title('Distribution of House Area')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()

# Mối quan hệ giữa diện tích và giá nhà
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Area', y='Price', data=df)
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Mối quan hệ giữa số phòng ngủ và giá nhà
plt.figure(figsize=(12, 6))
sns.barplot(x='No. of Bedrooms', y='Price', data=df)
plt.title('Average House Price by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x='Area', y='Price', data=df)
plt.title('House Price by Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


sample_data = df.sample(n=1000, random_state=42)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Location', y='Price', data=sample_data)
plt.title('House Price Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x='Gymnasium', y='Price', data=df)
plt.title('Average House Price by Gymnasium')
plt.xlabel('Gymnasium')
plt.ylabel('Average Price')
plt.show()
