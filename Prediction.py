import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Housing Price Dataset/train.csv/train.csv")
data_test = pd.read_csv("Housing Price Dataset/test.csv")
df.replace(9, np.nan, inplace=True)
df['No. of Bedrooms'] = df['No. of Bedrooms'].fillna(
    df['No. of Bedrooms'].mean())

amenity_columns = ['MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack', 'RainWaterHarvesting',
                   'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security',
                   'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
                   'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave',
                   'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']
df[amenity_columns] = df[amenity_columns].fillna(0)
# Xóa các cột không cần
df = df.drop(columns=['id'])
df = df.drop(columns=['Resale'])
df = df.drop(columns=['Location'])
# print(data_test.shape)

# print(df.describe())
# Tạo các đặc trưng mới


def mix_feature(df, train_data=True):
    df['Area_per_Bedrooms'] = df['Area'] / df['No. of Bedrooms']
    df['Bedrooms_Gymnasium'] = df['No. of Bedrooms'] * df['Gymnasium']
    # Biến đổi log và root
    if train_data:
        df['Log_Price'] = df['Price'].apply(
            lambda x: np.log(x) if x > 0 else 0)
    df['Root_Area'] = df['Area'].apply(lambda x: np.sqrt(x))

    # Kết hợp các biến tương tác khác:
    df['Staff_per_CarParking'] = df['MaintenanceStaff'] / \
        (df['CarParking'] + 1)
    df['Pool_Gym'] = df['SwimmingPool'] + df['Gymnasium']  # Fixed indentation
    df['Power_Security'] = df['PowerBackup'] + df['24X7Security']
    df['Sports_Club'] = df['SportsFacility'] * df['ClubHouse']
    df['Intercom_Vaastu'] = df['Intercom'] * df['VaastuCompliant']
    df['ATM_per_CarParking'] = df['ATM'] / (df['CarParking'] + 1)
    df['Sports_Pool'] = df['SportsFacility'] + df['SwimmingPool']
    df['Club_Gym'] = df['ClubHouse'] * df['Gymnasium']

    return df


# Data test
data_test = data_test.drop(columns=['id'])
data_test = data_test.drop(columns=['Resale'])
data_test = data_test.drop(columns=['Location'])
# mix feature
data_test = mix_feature(data_test, False)
df = mix_feature(df, True)
df = df.drop(columns=['Price'])


X = df.drop(columns=['Log_Price'])
y = df['Log_Price']

# data Preprocessing
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

# Mã hóa các biến phân loại
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical_X = encoder.fit_transform(X[categorical_columns])
encoded_categorical_data_test = encoder.transform(
    data_test[categorical_columns])

# Chuyển đổi kết quả mã hóa thành DataFrame
encoded_categorical_df_train = pd.DataFrame(
    encoded_categorical_X, columns=encoder.get_feature_names_out(categorical_columns))
encoded_categorical_df_test = pd.DataFrame(
    encoded_categorical_data_test, columns=encoder.get_feature_names_out(categorical_columns))

# Kết hợp các biến số và các biến phân loại đã mã hóa
data_train_encoded = pd.concat([X[numerical_columns], encoded_categorical_df_train], axis=1)
data_test_encoded = pd.concat([data_test[numerical_columns], encoded_categorical_df_test], axis=1)

# Chuẩn hóa data
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(data_train_encoded)
scaled_data_test = scaler.transform(data_test_encoded)

# Chuyển đổi data đã chuẩn hóa thành DataFrame
data_train_scaled = pd.DataFrame(
    scaled_data_train, columns=data_train_encoded.columns)
data_test_scaled = pd.DataFrame(
    scaled_data_test, columns=data_test_encoded.columns)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Lựa chọn biến quan trọng bằng RF


def select_features(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return X_train.columns[indices], importances[indices]


important_features_rf, importances_rf = select_features(
    X_train, y_train)
rf_feature_importance = pd.DataFrame(
    {'Feature': important_features_rf, 'Importance': importances_rf})
# print(rf_feature_importance)

# Lấy các feature trên 0.002
num_rf_features = len(rf_feature_importance)
important_features_rf_selected = important_features_rf[:num_rf_features]

X_train_selected = X_train[important_features_rf_selected]
X_test_selected = X_test[important_features_rf_selected]

# Huấn luyện mô hình Linear regression
linear_regressor_rf = LinearRegression()
linear_regressor_rf.fit(X_train_selected, y_train)

y_pred = linear_regressor_rf.predict(X_test_selected)

# Tính RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f'RMSE: {rmse}')

data_test_selected = data_test[important_features_rf_selected]
y_test_pred = linear_regressor_rf.predict(data_test_selected)
y_test_pred[:10]
submit_df = pd.read_csv('Housing Price Dataset/test.csv')
list_id = submit_df['id']
list_id = np.array(list_id.to_list())
len(list_id)
df = pd.DataFrame({'id': list_id, 'Price': y_test_pred})

print(df.to_csv('submission.csv', index=False))