#!/usr/bin/env python3
"""
Database setup script for HealthScope application
Run this before starting the main application
"""

import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import os
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "root"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "charset": "utf8mb4"
}

ADMIN_USERNAME = "admin12"
ADMIN_PASSWORD = "230406"

def setup_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect without specifying database
        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"]
        )
        
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS diapredict")
        print("✓ Database 'diapredict' created/verified")
        
        # Use the database
        cursor.execute("USE diapredict")
        
        # Create tables
        cursor.execute("""CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('user','admin') NOT NULL DEFAULT 'user'
        )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            input_data JSON NOT NULL,
            prediction TINYINT NOT NULL,
            probability FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        
        # Insert default admin if not exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (ADMIN_USERNAME,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'admin')",
                (ADMIN_USERNAME, generate_password_hash(ADMIN_PASSWORD))
            )
            print(f"✓ Admin user '{ADMIN_USERNAME}' created")
        else:
            print(f"✓ Admin user '{ADMIN_USERNAME}' already exists")

        connection.commit()
        print("✓ Tables created/verified")
        
        cursor.close()
        connection.close()
        
        print("✓ Database setup completed successfully!")
        return True
        
    except Error as e:
        print(f"✗ Database setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MySQL server is running")
        print("2. Check your MySQL credentials in this script")
        print("3. Ensure you have permission to create databases")
        return False

if __name__ == "__main__":
    print("Setting up HealthScope database...")
    setup_database()