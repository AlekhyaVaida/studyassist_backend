"""Database connection and setup."""

from motor.motor_asyncio import AsyncIOMotorClient
from src.config import config


class Database:
    client: AsyncIOMotorClient = None
    database = None


db = Database()


async def connect_to_mongo():
    """Create database connection."""
    db.client = AsyncIOMotorClient(config.mongodb_url)
    db.database = db.client[config.mongodb_db_name]
    print(f"Connected to MongoDB: {config.mongodb_db_name}")


async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        db.client.close()
        print("Disconnected from MongoDB")


async def get_database():
    """Get database instance."""
    return db.database

