import sqlite3

DB_NAME = "email_app.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            subject TEXT,
            body TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def connect():
    return sqlite3.connect(DB_NAME)


def create_user(username, password):
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username,password) VALUES(?,?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def login_user(username, password):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()
    conn.close()
    return user


def save_history(user_id, subject, body, prediction, confidence):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history(user_id, subject, body, prediction, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, subject, body, prediction, confidence))
    conn.commit()
    conn.close()


def fetch_history(user_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT subject, body, prediction, confidence, timestamp
        FROM history WHERE user_id=? ORDER BY timestamp DESC
    """, (user_id,))
    results = cur.fetchall()
    conn.close()
    return results

def fetch_history_all(user_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT subject, body, prediction, confidence, timestamp
        FROM history WHERE user_id=? ORDER BY timestamp DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows
