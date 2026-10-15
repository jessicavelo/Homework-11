import pandas as pd
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder \
    .appName("Homework_Grading_Demo") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Suppress noisy INFO logs so your print statements stand out
spark.sparkContext.setLogLevel("ERROR")

def log_step(step_num, title):
    print(f"\n{'='*20} STEP {step_num}: {title} {'='*20}")

# --- 1. Dataset Loading & Exploration ---
log_step(1, "Dataset Loading & Exploration")
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes") 
df_pd = pd.DataFrame(dataset['train'])
spark_df = spark.createDataFrame(df_pd)

print(f"Check 1.1: Dataset Loaded. Total rows: {spark_df.count()}")
print("Check 1.2: Identifying text + label fields:")
spark_df.printSchema()
spark_df.show(5)

# Authoritative 0/1 meaning, straight from the HuggingFace ClassLabel feature
print("\nCheck 1.3: Label encoding reported by the dataset itself:")
label_names = dataset['train'].features['label'].names
for i, name in enumerate(label_names):
    print(f"  label {i} = {name}")
    
# Class balance (rotten_tomatoes should be a perfect 4265 / 4265)
print("\nCheck 1.4: Class balance:")
spark_df.groupBy("label").count().orderBy("label").show()

# Data-quality guards
print("Check 1.5: Data quality:")
print(f"  distinct labels: {sorted(r.label for r in spark_df.select('label').distinct().collect())}")
print(f"  null labels    : {spark_df.filter(spark_df.label.isNull()).count()}")
print(f"  null/empty text: {spark_df.filter(spark_df.text.isNull() | (spark_df.text == '')).count()}")

# Eyeball both classes: the rows are stored sorted by label, so show(5) above
# only ever reveals one class. This is the check that proves the mapping.
print("\nCheck 1.6: Sample rows from each class:")
print("--- label = 1 (expect positive) ---")
spark_df.filter(spark_df.label == 1).select("text").show(3, truncate=90)
print("--- label = 0 (expect negative) ---")
spark_df.filter(spark_df.label == 0).select("text").show(3, truncate=90)

# --- 2. Data Preprocessing in Spark ---
log_step(2, "Data Preprocessing in Spark")

print("Check 2.1: DataFrame converted to Spark (schema verified)")
print(f"  Total rows: {spark_df.count()} | Columns: text (string), label (long)")
print("\n  Raw input sample:")
spark_df.select("text", "label").limit(1).show(truncate=70)

# Define stages
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+", toLowercase=True)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Demo on a single sample row
sample = spark.createDataFrame([("This movie is absolutely brilliant and fantastic!",)], ["text"])

print("\nCheck 2.2: TOKENIZATION - text → tokens")
step2_tokenized = tokenizer.transform(sample)
print("  Output: words column with tokens")
step2_tokenized.select("words").show(truncate=False)

print("\nCheck 2.3: STOPWORD REMOVAL - remove common words (the, is, and, etc.)")
step2_removed = remover.transform(step2_tokenized)
print("  Output: filtered column (stopwords removed)")
step2_removed.select("words", "filtered").show(truncate=False)

print("\nCheck 2.4: TF-IDF FEATURE EXTRACTION - convert to sparse vectors")
step2_hashed = hashingTF.transform(step2_removed)
print("  Output: rawFeatures (sparse vector with 5000 dimensions)")
step2_hashed.select("rawFeatures").show(truncate=False)

# Train IDF on full data, then transform sample
print("\nCheck 2.5: IDF SCALING - weight the features")
idf_model = idf.fit(hashingTF.transform(remover.transform(tokenizer.transform(spark_df))))
step2_idf = idf_model.transform(step2_hashed)
print("  Output: features (weighted TF-IDF sparse vector)")
step2_idf.select("features").show(truncate=False)

print("\nCheck 2.6: LABEL COLUMN - numeric classification (0=negative, 1=positive)")
print("  Sample labels from dataset:")
spark_df.select("label", "text").filter(spark_df.label == 1).limit(1).show(truncate=60)
spark_df.select("label", "text").filter(spark_df.label == 0).limit(1).show(truncate=60)

# --- 3. Model Training & Evaluation ---
log_step(3, "Model Training & Evaluation")
# Train/test split
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
print(f"Check 3.1: Split used. Train: {train_data.count()}, Test: {test_data.count()}")

# Classifier
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# Fit model
model = pipeline.fit(train_data)
print("Check 3.2: Model trained successfully.")

# Evaluation
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Check 3.3: Evaluation Metric (Accuracy): {accuracy:.4f}")

# --- 4. Model Saving & Inference ---
log_step(4, "Model Saving & Inference")
model_path = "spark_mlp_model"
model.write().overwrite().save(model_path)
print(f"Check 4.1: Saved trained pipeline to '{model_path}'")

# Load and run sample prediction
from pyspark.ml import PipelineModel
print(f"Load the saved model")
loaded_model = PipelineModel.load(model_path)
sample_text = spark.createDataFrame([("Brilliant and well-acted!",)], ["text"])
sample_pred = loaded_model.transform(sample_text)

print("Check 4.2: Inference on sample text:")
sample_pred.select("text", "prediction").show()

print("\n" + "*"*50)
print("DEMO LOGS COMPLETE")
print("*"*50)