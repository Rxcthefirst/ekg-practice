import psycopg2

conn = psycopg2.connect("dbname=mydb user=myuser password=mypass host=localhost port=5432")
cur = conn.cursor()
cur.execute("SELECT 1;")
print("Connection successful!" if cur.fetchone() else "Connection failed.")
cur.close()
conn.close()
