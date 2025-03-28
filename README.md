# Vectorizing RDF Data

This application is meant to demonstrate vectorizing RDF data for loading embeddings into PG vector for similarity search.


pip install rdflib psycopg2-binary sentence-transformers numpy

## Start the docker container from the yaml compose file
Execute the following command from the root directory:

<code>
docker compose up -d
</code>

## Connect to postgres DB
<code>
docker exec -it my_postgres psql -U myuser -d mydb
</code>

## Verify PG Vector is installed and usable
<code>
SELECT * FROM pg_extension WHERE extname = 'vector';
</code>

### Exit Postgres Terminal
Type <code>quit</code> in the terminal to exit the postgres command terminal.

## Check installed version of pg vector
<code>
SELECT * FROM pg_available_extensions WHERE name = 'vector';
</code>

Note: If "installed_version" field is blank, then the version is available but has not been installed for use.
#   e k g - p r a c t i c e  
 