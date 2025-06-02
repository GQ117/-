import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import os
from datetime import datetime

# 基础配置
N_SPLITS = 5  # 交叉验证折数
SEED = 42
DATA_PATH = '.'


def load_data():
    """加载训练和测试数据"""
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_PATH, 'testA.csv'))
    return train, test


def check_inf_and_nan(data, features):
    """检查并处理数据中的无穷大值和NaN值"""
    for col in features:
        if data[col].dtype in [np.float64, np.int64]:
            # 检查无穷大值
            inf_count = np.isinf(data[col]).sum()
            # 检查NaN值
            nan_count = np.isnan(data[col]).sum()

            if inf_count > 0 or nan_count > 0:
                print(f"特征 {col}: 无穷大值 = {inf_count}, NaN值 = {nan_count}")

                # 处理无穷大值
                if inf_count > 0:
                    # 将正无穷大替换为该特征的99.9%分位数
                    upper_bound = data[col][~np.isinf(data[col])].quantile(0.999)
                    data.loc[np.isinf(data[col]), col] = upper_bound

                    # 将负无穷大替换为该特征的0.1%分位数
                    lower_bound = data[col][~np.isinf(data[col])].quantile(0.001)
                    data.loc[np.isneginf(data[col]), col] = lower_bound

                # 处理NaN值
                if nan_count > 0:
                    # 用中位数填充NaN值
                    median_val = data[col].median()
                    data.loc[np.isnan(data[col]), col] = median_val
    return data


def preprocess_data(train, test):
    """数据预处理和特征工程"""
    # 提取目标变量
    y_train = train['isDefault']

    # 合并数据进行特征工程
    all_data = pd.concat([train.drop('isDefault', axis=1), test], axis=0).copy()  # 添加.copy()避免链式赋值警告

    # 处理日期特征
    if 'issueDate' in all_data.columns:
        all_data['issueDate'] = pd.to_datetime(all_data['issueDate'])
        all_data['issue_month'] = all_data['issueDate'].dt.month
        all_data['issue_dayofweek'] = all_data['issueDate'].dt.dayofweek

    if 'earliesCreditLine' in all_data.columns:
        # 指定日期格式以避免警告
        all_data['earliesCreditLine'] = pd.to_datetime(all_data['earliesCreditLine'], errors='coerce')
        all_data['credit_history_length'] = (all_data['issueDate'] - all_data['earliesCreditLine']).dt.days / 365.25

    # 处理employmentLength
    if 'employmentLength' in all_data.columns:
        # 使用非inplace方式避免FutureWarning
        all_data['employmentLength'] = all_data['employmentLength'].replace(to_replace='10+ years', value='10 years')
        all_data['employmentLength'] = all_data['employmentLength'].replace('< 1 year', '0 years')
        # 使用原始字符串避免警告
        all_data['employmentLength'] = all_data['employmentLength'].str.extract(r'(\d+)').astype(float)

    # 简单特征
    if 'loanAmnt' in all_data.columns and 'annualIncome' in all_data.columns:
        # 处理年收入为0的情况，避免除以零
        mask = all_data['annualIncome'] > 0
        all_data['loan_to_income'] = np.zeros(len(all_data))
        all_data.loc[mask, 'loan_to_income'] = all_data.loc[mask, 'loanAmnt'] / all_data.loc[mask, 'annualIncome']
        # 对于年收入为0的情况，设置一个极大值
        all_data.loc[~mask, 'loan_to_income'] = 999999999

    if 'ficoRangeLow' in all_data.columns and 'ficoRangeHigh' in all_data.columns:
        all_data['fico_score'] = (all_data['ficoRangeLow'] + all_data['ficoRangeHigh']) / 2

    # 分类特征频率编码
    categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        freq = all_data[col].value_counts(normalize=True)
        all_data[f'{col}_freq'] = all_data[col].map(freq)

    # 检查并处理无穷大值和NaN值
    numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in ['id']]
    all_data = check_inf_and_nan(all_data, numerical_features)

    # 分离训练集和测试集
    train_size = len(train)
    test_id = test['id']

    # 预处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 应用预处理
    all_data_processed = preprocessor.fit_transform(all_data)

    X_train = all_data_processed[:train_size]
    X_test = all_data_processed[train_size:]

    return X_train, y_train, X_test, test_id


def train_model(X_train, y_train):
    """使用LightGBM进行交叉验证训练"""
    models = []
    oof_preds = np.zeros(len(y_train))

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': SEED,
        'n_jobs': -1
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"开始训练第 {fold + 1}/{N_SPLITS} 折...")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        # 使用回调函数实现早停，兼容旧版本LightGBM
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=callbacks,  # 使用回调函数
        )

        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
        models.append(model)
        print(f"第 {fold + 1}/{N_SPLITS} 折训练完成")

    return models, oof_preds


def predict_and_save(models, X_test, test_id):
    """生成预测结果并保存"""
    # 集成预测
    all_preds = [model.predict(X_test) for model in models]
    final_preds = np.mean(all_preds, axis=0)

    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_id,
        'isDefault': final_preds
    })

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission.to_csv(f'submission_{timestamp}.csv', index=False)
    return submission


def main():
    """主函数"""
    print("开始金融贷款违约预测...")

    # 加载数据
    print("加载数据...")
    train, test = load_data()

    # 数据预处理
    print("数据预处理...")
    X_train, y_train, X_test, test_id = preprocess_data(train, test)

    # 模型训练
    print("模型训练...")
    models, oof_preds = train_model(X_train, y_train)

    # 生成预测结果
    print("生成预测结果...")
    submission = predict_and_save(models, X_test, test_id)

    print("预测完成!")


if __name__ == "__main__":
    main()