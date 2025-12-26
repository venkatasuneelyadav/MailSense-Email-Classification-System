import sqlite3
from database import get_db_connection

def create_user(username, password):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users(username, password) VALUES (?, ?)", 
                    (username, password))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def login_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?", 
                (username, password))
    user = cur.fetchone()
    conn.close()
    return user
