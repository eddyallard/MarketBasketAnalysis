import time

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

# On calcule le temps que prend le programme à s'éxécuter avec time.
start_time = time.time()
# Constants
spark_master = 'spark://spark-master:7077'
hdfs_input_folder = 'hdfs://namenode:9000/inputs/'
app_name = 'MBA'

spark = SparkSession.Builder().master(spark_master)\
    .appName(app_name)\
    .getOrCreate()

order_products_train = spark.read\
    .option('header', 'true')\
    .csv(f'{hdfs_input_folder}order_products__train.csv')

print(f"order_products_train count = {order_products_train.count()}")

order_products_prior = spark.read\
    .option('header', 'true')\
    .csv(f'{hdfs_input_folder}order_products__prior.csv')
#   Données sur les produits seront chargées après que les traitements soient faits pour tenter d'économiser un peu
#   de mémoire.
print(f"order_products_prior count = {order_products_prior.count()}")
#   Nous pouvons combiner nos 2 dataframes sans problème car ils ont tous deux le même format:
#   (order_id, product_id, add_to_cart_order, reordered)
#   Cependant, tout ce qui nous intéresse est le order_id et le product_id alors nous allons devoir
#   modifier nos données.
baskets = order_products_prior.union(order_products_train)
#   On libère les dataframes qui ne seront plus nécéssaires.
order_products_prior.unpersist()
order_products_train.unpersist()
print(f"order_products count = {baskets.count()}")
#   Nous allons maintenant changer les ID pour les noms de produits.
products = spark.read\
    .option('header', 'true')\
    .csv(f'{hdfs_input_folder}products.csv')
baskets = baskets.join(products, baskets.product_id == products.product_id)
products.unpersist()
#   Ici nous combinons toutes les rows sur ayant le même numéro de commande, et nous créons une liste
#   de produits. De cette façon, nous rejetons les colonnes que nous n'utilisions pas, et nous avons
#   une représentation de chaque panier.
baskets = baskets.groupby('order_id')\
    .agg(collect_list('product_name').alias('products'))

print(f"grouped count = {baskets.count()}")
baskets.show(5, truncate=False)
#   -------------------------------------------
#   Trouver les motifs fréquents avec FP-Growth.
#   -------------------------------------------
fp_growth = FPGrowth()\
    .setItemsCol('products')\
    .setMinSupport(0.01)\
    .setMinConfidence(0.125)
#   min support = 0.01 alors le produit doit apparaitre dans 1% des commandes pour être considéré fréquent
#   min confidence = pour qu'une association soit faite, dans x => y, y doit être dans au moins 12,5% des commandes
#   où x se trouve.
fpm_model = fp_growth.fit(baskets)
fpm_model.setPredictionCol('new_prediction')

frequent_itemsets = fpm_model.freqItemsets
frequent_itemsets.createOrReplaceTempView('frequent')
query = 'select * ' \
        'from frequent ' \
        'order by freq desc ' \
        'limit 10'
print(f"Voici les 10 itemsets les plus pertinants...\n")
frequent_itemsets = spark.sql(query)
frequent_itemsets.show(truncate=False)
#   -------------------------------------------
#   Trouver les règles d'association
#   -------------------------------------------
#   fpm_model.associationRules nous retourne un dataframe contenant les regles d'association
association_rules = fpm_model.associationRules
association_rules.show(5, truncate=False)
association_rules.createOrReplaceTempView('association_rules')
#   nous pouvons avoir le lift et confidence de (antecedent -> consequent), grâce à model.associationRules.
query = 'select * ' \
        'from association_rules ' \
        'order by confidence desc ' \
        'limit 10'
top_rules = spark.sql(query)
print(f"Voici les 10 règles les plus pertinantes...\n")
top_rules.show(truncate=False)
#   -------------------------------------------
#   Évaluer le modèle d'apprentissage fpm_model
#   -------------------------------------------
#   La prédiction que nous feront, va nous montrer ce que la personne achetant dans ce cas, un citron
#   et des fraises, serait portée à ajouter à son panier.
test_transaction = [(0, ['Large Lemon', 'Strawberries'])]
test_df = spark.createDataFrame(test_transaction, ["order_id", "products"])
fpm_model.transform(test_df).show(truncate=False)

print(f"Le programme a pris {time.time()-start_time} secondes à s'éxécuter.")
spark.stop()
