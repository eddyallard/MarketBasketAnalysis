import time

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

spark = SparkSession.Builder().master('spark://spark-master:7077')\
    .appName('MBA')\
    .getOrCreate()

order_products_train = spark.read\
    .option('header', 'true')\
    .csv('hdfs://namenode:9000/inputs/order_products__train.csv')

print(f"order_products_train count = {order_products_train.count()}")

order_products_prior = spark.read\
    .option('header', 'true')\
    .csv('hdfs://namenode:9000/inputs/order_products__prior.csv')
#   Données sur les produits seront chargées après que les traitements soient faits pour tenter d'économiser un peu
#   de mémoire.
print(f"order_products_prior count = {order_products_prior.count()}")
#   Nous pouvons combiner nos 2 dataframes sans problème car ils ont tous deux le même format:
#   (order_id, product_id, add_to_cart_order, reordered)
#   Cependant, tout ce qui nous intéresse est le order_id et le product_id alors nous allons devoir
#   modifier nos données.
order_products = order_products_prior.union(order_products_train)
#   On libère les dataframes qui ne seront plus nécéssaires.
order_products_prior.unpersist()
order_products_train.unpersist()
print(f"order_products count = {order_products.count()}")
#   Nous allons maintenant changer les ID pour les noms de produits.
products = spark.read\
    .option('header', 'true')\
    .csv('hdfs://namenode:9000/inputs/products.csv')
order_products = order_products.join(products, order_products.product_id == products.product_id)
products.unpersist()
#   Ici nous combinons toutes les rows sur ayant le même numéro de commande, et nous créons une liste
#   de produits. De cette façon, nous rejetons les colonnes que nous n'utilisions pas, et nous avons
#   une représentation de chaque panier.
order_products = order_products.groupby('order_id')\
    .agg(collect_list('product_name').alias('products'))

print(f"grouped count = {order_products.count()}")
order_products.show(5,truncate=False)
#   -------------------------------------------
#   Trouver les motifs fréquents avec FP-Growth
#   -------------------------------------------
#   On utilise time pour calculer le temps que l'opération de fit prend.
start_time = time.time()
fp_growth = FPGrowth()\
    .setItemsCol('products')\
    .setMinSupport(0.01)\
    .setMinConfidence(0.25)
#   min support = 0.01 alors le produit doit apparaitre dans 1% des commandes pour être considéré fréquent
#   min confidence = pour qu'une association soit faite, dans x => y, y doit être dans au moins 25% des commandes
#   où x se trouve.
fpm_model = fp_growth.fit(order_products)
fpm_model.setPredictionCol('new_prediction')
print(f"fit costs: {time.time()-start_time} seconds")

most_popular = fpm_model.freqItemsets
most_popular.show(10, truncate=False)
spark.stop()
