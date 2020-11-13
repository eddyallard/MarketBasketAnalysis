from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

spark = SparkSession.Builder().master('spark://spark-master:7077')\
    .appName('MBA')\
    .getOrCreate()

order_products_train = spark.read\
    .option('header', 'true')\
    .csv('hdfs://namenode:9000/inputs/order_products__train.csv')

print(f"First count ={order_products_train.count()}")

order_products_prior = spark.read\
    .option('header', 'true')\
    .csv('hdfs://namenode:9000/inputs/order_products__prior.csv')
# Données sur les produits seront chargées après que les traitements soient faits.
print(f"Second count ={order_products_prior.count()}")
# Nous pouvons combiner nos 2 dataframes sans problème car ils ont tous deux le même format:
# (order_id, product_id, add_to_cart_order, reordered)
# Cependant, tout ce qui nous intéresse est le order_id et le product_id alors nous allons devoir
# modifier nos données.
order_products = order_products_prior.union(order_products_train)

print(f"Combined count ={order_products.count()}")
# Ici nous combinons toutes les rows sur ayant le même numéro de commande, et nous créons une liste
# de produits. De cette façon, nous rejetons les colonnes que nous n'utilisions pas, et nous avons
# une représentation de chaque panier.
order_products = order_products.groupby('order_id')\
    .agg(collect_list("product_id").alias("product_id"))

print(f"Grouped count ={order_products.count()}")
order_products.show(10,truncate=False)
spark.stop()
