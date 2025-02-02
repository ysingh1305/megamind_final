# dbauthentication.py

import sqlite3

DATABASE = "database.db"

def get_connection():
    """Create and return a connection to the SQLite database."""
    return sqlite3.connect(DATABASE)

def create_tables():
    """Create the users, quiz_stats, and quiz_history tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table to store user credentials.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    # Table to store aggregated quiz statistics.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quiz_stats (
            username TEXT PRIMARY KEY,
            total_quizzes INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            total_correct INTEGER DEFAULT 0,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    # Table to store each individual quiz attempt.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            quiz_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_questions INTEGER,
            score INTEGER,
            percentage REAL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username, email, password):
    """
    Add a new user to the database.
    
    Returns:
        True if the user was added successfully,
        False if the username or email already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Insert user credentials into the 'users' table.
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password)
        )
        # Create an initial record in quiz_stats.
        cursor.execute(
            "INSERT INTO quiz_stats (username, total_quizzes, total_questions, total_correct) VALUES (?, 0, 0, 0)",
            (username,)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """
    Verify a user's login credentials.
    
    Returns:
        True if the username exists and the password matches,
        False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        stored_password = row[0]
        # In production, compare hashed passwords.
        return password == stored_password
    return False

def update_quiz_stats(username, quiz_count, total_questions, score, percentage):
    """
    Update the quiz statistics for a user and log the quiz attempt.
    
    Args:
        username (str): The username.
        quiz_count (int): The number of quizzes to add (usually 1).
        total_questions (int): The total number of questions in the quiz.
        score (int): The number of correct answers.
        percentage (float): The score percentage for this quiz.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Update aggregated quiz_stats.
    cursor.execute(
        "SELECT total_quizzes, total_questions, total_correct FROM quiz_stats WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    
    if row:
        new_total_quizzes = row[0] + quiz_count
        new_total_questions = row[1] + total_questions
        new_total_correct = row[2] + score
        cursor.execute(
            "UPDATE quiz_stats SET total_quizzes = ?, total_questions = ?, total_correct = ? WHERE username = ?",
            (new_total_quizzes, new_total_questions, new_total_correct, username)
        )
    else:
        cursor.execute(
            "INSERT INTO quiz_stats (username, total_quizzes, total_questions, total_correct) VALUES (?, ?, ?, ?)",
            (username, quiz_count, total_questions, score)
        )
    
    # Insert a record into quiz_history for this quiz attempt.
    cursor.execute(
        "INSERT INTO quiz_history (username, total_questions, score, percentage) VALUES (?, ?, ?, ?)",
        (username, total_questions, score, percentage)
    )
    
    conn.commit()
    conn.close()

def get_user_stats(username):
    """
    Retrieve the aggregated quiz statistics for a user.
    
    Returns:
        A dictionary with keys: 'total_quizzes', 'total_questions', 'total_correct', 'average_percentage'
        or None if no stats are found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT total_quizzes, total_questions, total_correct FROM quiz_stats WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        total_quizzes, total_questions, total_correct = row
        average_percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0
        return {
            "total_quizzes": total_quizzes,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "average_percentage": average_percentage
        }
    return None

def get_quiz_history(username):
    """
    Retrieve the quiz history for a user.
    
    Returns:
        A list of tuples: (id, username, quiz_date, total_questions, score, percentage)
        sorted by quiz_date in ascending order.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, quiz_date, total_questions, score, percentage FROM quiz_history WHERE username = ? ORDER BY quiz_date ASC",
        (username,)
    )
    records = cursor.fetchall()
    conn.close()
    return records
