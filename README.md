import json
import jieba
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import Levenshtein
import re
import joblib
import os

# 初始化jieba分词器
jieba.initialize()

# 特征提取函数
def extract_features(q1, q2):
    features = []

    # 1. 文本长度特征
    len_q1 = len(q1)
    len_q2 = len(q2)
    features.append(len_q1)
    features.append(len_q2)
    features.append(abs(len_q1 - len_q2))
    features.append(min(len_q1, len_q2) / (max(len_q1, len_q2) + 1e-8))  # 防止除以零

    # 2. 分词相关特征
    tokens1 = list(jieba.cut(q1))
    tokens2 = list(jieba.cut(q2))
    set1 = set(tokens1)
    set2 = set(tokens2)

    # 词数量特征
    word_count1 = len(tokens1)
    word_count2 = len(tokens2)
    features.append(word_count1)
    features.append(word_count2)
    features.append(abs(word_count1 - word_count2))

    # 3. 词重叠特征
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    # Jaccard相似度
    jaccard = intersection / union if union > 0 else 0
    features.append(jaccard)

    # Dice系数
    dice = 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
    features.append(dice)

    # 4. 编辑距离特征
    edit_distance = Levenshtein.distance(q1, q2)
    features.append(edit_distance)
    # 归一化编辑距离
    max_len = max(len_q1, len_q2)
    norm_edit_distance = edit_distance / max_len if max_len > 0 else 1
    features.append(norm_edit_distance)

    # 5. 相同字符比例
    same_chars = sum(1 for c in q1 if c in q2)
    char_sim = same_chars / max_len if max_len > 0 else 0
    features.append(char_sim)

    # 6. 数字特征
    nums1 = re.findall(r'\d+', q1)
    nums2 = re.findall(r'\d+', q2)
    num_match = int(set(nums1) == set(nums2))
    features.append(num_match)

    return features


# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 标签处理函数（修复NA问题）
def process_label(label):
    """安全地处理标签值，处理特殊情况"""
    # 如果标签是字符串表示的整数
    if isinstance(label, str) and label.isdigit():
        return int(label)
    # 如果标签已经是整数
    elif isinstance(label, int):
        return label
    # 处理"NA"或其他非数值标签
    else:
        # 尝试转换为整数（针对字符串类型的数字）
        try:
            return int(label)
        except (ValueError, TypeError):
            # 返回默认值0
            return 0


# 主函数
def main():
    # 检查模型文件是否存在
    model_exists = os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl')

    # 1. 加载数据集
    train_data = load_data('KUAKE-QQR_train.json')
    test_data = load_data('KUAKE-QQR_test.json')

    # 2. 提取测试集特征
    X_test = []
    test_ids = []
    for sample in test_data:
        features = extract_features(sample['query1'], sample['query2'])
        X_test.append(features)
        test_ids.append(sample['id'])

    if not model_exists:
        print("训练模式：训练新模型并预测测试集")
        # 3. 加载训练集（使用改进的标签处理）
        X_train, y_train = [], []
        for sample in train_data:
            features = extract_features(sample['query1'], sample['query2'])
            X_train.append(features)
            # 使用改进的标签处理函数
            y_train.append(process_label(sample.get('label', 0)))

        # 4. 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. 模型训练
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }

        best_model = None
        best_model_name = ''

        if os.path.exists('KUAKE-QQR_dev.json'):
            print("使用验证集选择最佳模型")
            dev_data = load_data('KUAKE-QQR_dev.json')
            X_dev, y_dev = [], []
            for sample in dev_data:
                features = extract_features(sample['query1'], sample['query2'])
                X_dev.append(features)
                # 使用改进的标签处理函数
                y_dev.append(process_label(sample.get('label', 0)))

            X_dev = scaler.transform(X_dev)
            best_acc = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_dev)
                acc = accuracy_score(y_dev, y_pred)
                print(f"{name} 验证集准确率: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_model_name = name

            print(f"\n选择最佳模型: {best_model_name}，准确率: {best_acc:.4f}")
        else:
            # 如果没有验证集，使用逻辑回归作为默认模型
            print("未找到验证集，使用逻辑回归作为默认模型")
            best_model = LogisticRegression(max_iter=1000)
            best_model.fit(X_train, y_train)
            best_model_name = 'LogisticRegression'

        # 保存模型和标准化器
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("已保存训练好的模型和标准化器")

        # 6. 测试集预测
        test_preds = best_model.predict(X_test_scaled)
    else:
        print("预测模式：加载已有模型预测测试集")
        # 加载已有模型和标准化器
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        test_preds = best_model.predict(X_test_scaled)
        print(f"使用模型: {type(best_model).__name__}")

    # 7. 保存结果（符合赛题要求）
    result_data = []
    for i, sample in enumerate(test_data):
        result_data.append({
            "id": sample["id"],
            "query1": sample["query1"],
            "query2": sample["query2"],
            "label": str(test_preds[i])
        })

    with open('KUAKE-QQR_test_pred.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
