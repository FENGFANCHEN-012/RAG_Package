"""
 ============================================================================
 意图分类器 (intent_classifier.py)
 ============================================================================
 
 功能描述：
 - 使用 TF-IDF + SVM 判断用户查询意图
 - 将用户查询分类为：database_query（数据库查询）、rag_query（RAG查询）、normal_query（普通对话）
 - 延迟低，适合实时场景
 - 支持模型训练和加载
 
 主要类：
 - IntentClassifier: 意图分类器类
   - __init__: 初始化分类器，尝试加载已训练模型
   - _preprocess_text: 文本预处理（jieba分词）
   - train: 训练分类器
   - predict: 预测查询意图
   - load: 加载已训练模型
 
 意图类别：
 - database_query: 数据库查询（员工、部门等）
 - rag_query: RAG 查询（包括医疗知识问答和普通对话） 普通对话
 
 依赖：
 - scikit-learn: TF-IDF 和 SVM
 - jieba: 中文分词
 
 ============================================================================
"""
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import jieba
import re

# 意图类别定义
INTENT_LABELS = ["database_query", "rag_query", "normal_query"]

class IntentClassifier:
    def __init__(self, model_path="./intent_model"):
        """
        初始化意图分类器
        model_path: 训练好的模型路径
        """
        self.model_path = model_path
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
        # 尝试加载已训练的模型
        if os.path.exists(model_path):
            self.load()
    
    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理：分词、去除特殊字符
        """
        # 使用 jieba 分词
        words = jieba.lcut(text)
        # 去除标点和特殊字符
        words = [w for w in words if re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', w)]
        return ' '.join(words)
    
    def train(self, training_data, output_dir="./intent_model"):
        """
        训练分类器
        training_data: 训练数据列表，格式 [{"text": "...", "label": "..."}, ...]
        output_dir: 模型保存路径
        """
        print(f"开始训练，数据量: {len(training_data)}")
        
        # 准备数据， label 和 text 分开
        texts = [self._preprocess_text(item["text"]) for item in training_data]
        labels = [INTENT_LABELS.index(item["label"]) for item in training_data]
        
        # 使用 TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # 使用 1-gram 和 2-gram
            min_df=1
        )
        X = self.vectorizer.fit_transform(texts)
        
        # 使用 SVM 分类器
        self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        self.classifier.fit(X, labels)
        
        self.is_trained = True
        
        # 保存模型
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(output_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"模型已保存到 {output_dir}")
        
        # 显示训练准确率
        predictions = self.classifier.predict(X)
        accuracy = sum(predictions == labels) / len(labels)
        print(f"训练准确率: {accuracy:.2%}")
    
    def predict(self, text: str) -> str:
        """
        预测文本的意图
        返回意图类别字符串
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用 train() 方法")
        
        # 预处理
        processed_text = self._preprocess_text(text)
        
        # 向量化
        X = self.vectorizer.transform([processed_text])
        
        # 预测
        predicted_class_id = self.classifier.predict(X)[0]
        
        return INTENT_LABELS[predicted_class_id]
    
    def load(self):
        """
        从文件加载已训练的模型
        """
        vectorizer_path = os.path.join(self.model_path, 'vectorizer.pkl')
        classifier_path = os.path.join(self.model_path, 'classifier.pkl')
        
        if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.is_trained = True
            print(f"模型已从 {self.model_path} 加载")
        else:
            print(f"未找到训练好的模型，需要先训练")


# 示例训练数据
SAMPLE_TRAINING_DATA = [
    # 数据库查询
    {"text": "查询所有员工", "label": "database_query"},
    {"text": "显示部门信息", "label": "database_query"},
    {"text": "统计员工数量", "label": "database_query"},
    {"text": "查找John的信息", "label": "database_query"},
    {"text": "列出所有部门", "label": "database_query"},
    {"text": "工程部邮箱是什么", "label": "database_query"},
    {"text": "有多少个员工", "label": "database_query"},
    {"text": "查询Marketing部门", "label": "database_query"},
    {"text": "员工工资是多少", "label": "database_query"},
    {"text": "哪个部门的人最多", "label": "database_query"},
    {"text": "显示所有经理", "label": "database_query"},
    {"text": "统计销售额", "label": "database_query"},
    {"text": "查找张三的联系方式", "label": "database_query"},
    {"text": "显示销售部门的员工", "label": "database_query"},
    {"text": "统计各部门人数", "label": "database_query"},
    
    # RAG 查询（包括医疗知识、公司政策、普通对话、财务文档查询）
    {"text": "糖尿病注意事项", "label": "rag_query"},
    {"text": "感冒怎么办", "label": "rag_query"},
    {"text": "高血压症状", "label": "rag_query"},
    # 财务资产报告相关查询 - 资产总览
    {"text": "公司总资产是多少", "label": "rag_query"},
    {"text": "2025财年总资产", "label": "rag_query"},
    {"text": "资产增长多少", "label": "rag_query"},
    {"text": "固定资产占比", "label": "rag_query"},
    {"text": "无形资产占比", "label": "rag_query"},
    {"text": "库存资产占比", "label": "rag_query"},
    {"text": "数字资产占比", "label": "rag_query"},
    # 不动产查询
    {"text": "上海总部建筑面积", "label": "rag_query"},
    {"text": "苏州工厂价值", "label": "rag_query"},
    {"text": "成都物流中心", "label": "rag_query"},
    {"text": "广州临床研究中心", "label": "rag_query"},
    {"text": "北京销售中心", "label": "rag_query"},
    {"text": "不动产账面价值", "label": "rag_query"},
    {"text": "公司有哪些房产", "label": "rag_query"},
    # 生产设备查询
    {"text": "冻干生产线", "label": "rag_query"},
    {"text": "无菌灌装线", "label": "rag_query"},
    {"text": "液相色谱仪", "label": "rag_query"},
    {"text": "质谱分析系统", "label": "rag_query"},
    {"text": "仓储系统", "label": "rag_query"},
    {"text": "苏州工厂设备", "label": "rag_query"},
    {"text": "生产设备价值", "label": "rag_query"},
    # 运输设备查询
    {"text": "冷链运输车", "label": "rag_query"},
    {"text": "移动药房终端", "label": "rag_query"},
    {"text": "运输车辆数量", "label": "rag_query"},
    # 知识产权查询
    {"text": "公司有多少项发明专利", "label": "rag_query"},
    {"text": "实用新型专利", "label": "rag_query"},
    {"text": "商标有哪些", "label": "rag_query"},
    {"text": "药品注册批件", "label": "rag_query"},
    {"text": "商业秘密", "label": "rag_query"},
    {"text": "知识产权价值", "label": "rag_query"},
    {"text": "核心化合物专利", "label": "rag_query"},
    # 软件系统查询
    {"text": "公司有哪些软件系统", "label": "rag_query"},
    {"text": "SAP系统", "label": "rag_query"},
    {"text": "Salesforce", "label": "rag_query"},
    {"text": "LIMS系统", "label": "rag_query"},
    {"text": "药物警戒系统", "label": "rag_query"},
    {"text": "EAP管理平台", "label": "rag_query"},
    # 许可证查询
    {"text": "公司有哪些许可证", "label": "rag_query"},
    {"text": "GMP认证", "label": "rag_query"},
    {"text": "药品经营许可证", "label": "rag_query"},
    {"text": "麻醉药品证明", "label": "rag_query"},
    {"text": "进出口药品资质", "label": "rag_query"},
    # 商誉查询
    {"text": "公司商誉是怎么产生的", "label": "rag_query"},
    {"text": "收购安宁生物", "label": "rag_query"},
    {"text": "商誉价值", "label": "rag_query"},
    # 库存资产查询
    {"text": "库存资产有哪些药品", "label": "rag_query"},
    {"text": "安息平库存数量", "label": "rag_query"},
    {"text": "静眠妥库存", "label": "rag_query"},
    {"text": "止息露库存", "label": "rag_query"},
    {"text": "忘川口服液库存", "label": "rag_query"},
    {"text": "惜别贴库存", "label": "rag_query"},
    {"text": "渡厄针库存", "label": "rag_query"},
    {"text": "在制品有哪些", "label": "rag_query"},
    {"text": "原材料库存", "label": "rag_query"},
    {"text": "近效期库存有哪些", "label": "rag_query"},
    {"text": "报废库存", "label": "rag_query"},
    # 数字资产查询
    {"text": "数字资产包括什么", "label": "rag_query"},
    {"text": "临床数据资产价值", "label": "rag_query"},
    {"text": "III期临床数据", "label": "rag_query"},
    {"text": "真实世界研究数据", "label": "rag_query"},
    {"text": "药物警戒数据库", "label": "rag_query"},
    {"text": "内部知识库", "label": "rag_query"},
    {"text": "患者教育内容库", "label": "rag_query"},
    {"text": "AI辅助诊断模型", "label": "rag_query"},
    {"text": "网站域名", "label": "rag_query"},
    {"text": "mortalitus.com", "label": "rag_query"},
    # 其他资产查询
    {"text": "预付账款", "label": "rag_query"},
    {"text": "临床试验预付", "label": "rag_query"},
    {"text": "设备预付款", "label": "rag_query"},
    {"text": "保证金与押金", "label": "rag_query"},
    {"text": "租房押金", "label": "rag_query"},
    {"text": "药品流通保证金", "label": "rag_query"},
    # 折旧政策查询
    {"text": "资产折旧政策是什么", "label": "rag_query"},
    {"text": "建筑物折旧年限", "label": "rag_query"},
    {"text": "生产设备折旧", "label": "rag_query"},
    {"text": "软件折旧", "label": "rag_query"},
    {"text": "专利摊销", "label": "rag_query"},
    # 风险查询
    {"text": "资产风险有哪些", "label": "rag_query"},
    {"text": "麻醉药品原料风险", "label": "rag_query"},
    {"text": "临床数据风险", "label": "rag_query"},
    {"text": "近效期库存风险", "label": "rag_query"},
    {"text": "AI模型合规", "label": "rag_query"},
    {"text": "苏州工厂环保", "label": "rag_query"},
    # 综合查询
    {"text": "MORTALITUS公司资产", "label": "rag_query"},
    {"text": "2025财年报告", "label": "rag_query"},
    {"text": "财务资产部报告", "label": "rag_query"},
    {"text": "公司资产结构", "label": "rag_query"},
    {"text": "胰岛素用法", "label": "rag_query"},
    {"text": "心脏病预防", "label": "rag_query"},
    {"text": "阿司匹林副作用", "label": "rag_query"},
    {"text": "血糖正常值", "label": "rag_query"},
    {"text": "疫苗接种时间", "label": "rag_query"},
    {"text": "公司礼品政策是什么", "label": "rag_query"},
    {"text": "员工丧假怎么申请", "label": "rag_query"},
    {"text": "患者死亡时如何沟通", "label": "rag_query"},
    {"text": "医疗合规政策有哪些", "label": "rag_query"},
    {"text": "你好", "label": "rag_query"},
    {"text": "谢谢", "label": "rag_query"},
    {"text": "再见", "label": "rag_query"},
    {"text": "你叫什么名字", "label": "rag_query"},
    {"text": "今天天气怎么样", "label": "rag_query"},
    {"text": "查询医疗知识", "label": "rag_query"},
    {"text": "公司有哪些规定", "label": "rag_query"},
    {"text": "如何处理违规请求", "label": "rag_query"},
    {"text": "什么是MTL-408", "label": "rag_query"},
    {"text": "医疗咨询流程是什么", "label": "rag_query"},
    {"text": "如何识别灰色地带", "label": "rag_query"},
    {"text": "员工行为规范有哪些", "label": "rag_query"},
    {"text": "殡葬服务推荐规定", "label": "rag_query"},
    {"text": "强制心理咨询政策", "label": "rag_query"},
    {"text": "礼品接收限制", "label": "rag_query"},
    {"text": "医疗道德政策", "label": "rag_query"},
    {"text": "什么是公司合规", "label": "rag_query"},
    {"text": "如何提供替代方案", "label": "rag_query"},
    {"text": "违规行为评分标准", "label": "rag_query"},
    {"text": "医疗文件要求", "label": "rag_query"},
    {"text": "心理咨询相关规定", "label": "rag_query"},
    {"text": "你是谁", "label": "rag_query"},
    {"text": "能帮我做什么", "label": "rag_query"},
    {"text": "介绍一下你自己", "label": "rag_query"},
    {"text": "今天几号", "label": "rag_query"},
    {"text": "现在几点了", "label": "rag_query"},
    {"text": "讲个笑话", "label": "rag_query"},
]


if __name__ == "__main__":
    # 使用示例
    classifier = IntentClassifier()
    
    # 训练（需要先收集足够的数据）
    print("开始训练...")
    classifier.train(SAMPLE_TRAINING_DATA, output_dir="./intent_model")
    
    # 测试
    print("\n测试分类器:")
    test_queries = [
        "查询员工信息",
        "糖尿病怎么治",
        "你好",
        "工程部邮箱",
        "今天天气真好",
        "安息平在库多少盒？"
    ]
    
    for query in test_queries:
        intent = classifier.predict(query)
        print(f"查询: {query} -> 意图: {intent}")
