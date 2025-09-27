import sqlite3

# Conectar a la base de datos (se creará si no existe)
conn = sqlite3.connect('data/database.db')
cursor = conn.cursor()

# Crear una tabla
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL
)
''')

# Insertar datos en la tabla
cursor.execute('INSERT INTO employees (name, age) VALUES (?, ?)', ("John Doe", 28))
cursor.execute('INSERT INTO employees (name, age) VALUES (?, ?)', ("Jane Smith", 34))
cursor.execute('INSERT INTO employees (name, age) VALUES (?, ?)', ("Emily Johnson", 45))

# Guardar los cambios y cerrar la conexión
conn.commit()
conn.close()
