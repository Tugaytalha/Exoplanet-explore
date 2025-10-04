#!/usr/bin/env python3
"""
MongoDB Population Script for Exoplanet Data

This script populates MongoDB with additional planet data that will be
merged with the API responses from the /planets endpoint.

Usage:
    python populate_mongodb.py
"""

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "exoplanet_db")

def connect_to_mongodb():
    """Connect to MongoDB and return collection."""
    try:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client[MONGODB_DB]
        collection = db['planets']
        print(f"✅ Connected to MongoDB at {MONGODB_URL}")
        print(f"   Database: {MONGODB_DB}")
        print(f"   Collection: planets")
        return collection
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ Could not connect to MongoDB: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def populate_sample_data(collection):
    """Populate MongoDB with sample planet data."""
    
    # Sample additional data for planets
    sample_planets = [
        {
            "kepid": 10797460,
            "planet_type": "Super Earth",
            "mass_earth": 5.2,
            "radius_earth": 1.8,
            "orbital_period_days": 12.3,
            "discovery_year": 2011,
            "discovery_method": "Transit",
            "host_star_type": "G-type",
            "temperature_kelvin": 450,
            "habitable_zone": False,
            "atmosphere_composition": ["H2", "He"],
            "notes": "Interesting planet with potential for further study"
        },
        {
            "kepid": 10811496,
            "planet_type": "Hot Jupiter",
            "mass_earth": 318.2,
            "radius_earth": 11.2,
            "orbital_period_days": 3.5,
            "discovery_year": 2011,
            "discovery_method": "Transit",
            "host_star_type": "F-type",
            "temperature_kelvin": 1200,
            "habitable_zone": False,
            "atmosphere_composition": ["H2", "He", "CH4"],
            "notes": "Gas giant orbiting very close to its host star"
        },
        {
            "kepid": 10848459,
            "planet_type": "Earth-like",
            "mass_earth": 1.1,
            "radius_earth": 1.05,
            "orbital_period_days": 365.0,
            "discovery_year": 2012,
            "discovery_method": "Transit",
            "host_star_type": "G-type",
            "temperature_kelvin": 288,
            "habitable_zone": True,
            "atmosphere_composition": ["N2", "O2", "Ar"],
            "notes": "Potentially habitable planet with Earth-like characteristics"
        },
        {
            "kepid": 10854555,
            "planet_type": "Neptune-like",
            "mass_earth": 17.1,
            "radius_earth": 3.9,
            "orbital_period_days": 45.2,
            "discovery_year": 2012,
            "discovery_method": "Transit",
            "host_star_type": "K-type",
            "temperature_kelvin": 320,
            "habitable_zone": False,
            "atmosphere_composition": ["H2", "He", "CH4"],
            "notes": "Ice giant with thick atmosphere"
        }
    ]
    
    try:
        # Clear existing data (optional)
        result = collection.delete_many({})
        print(f"\n🗑️  Cleared {result.deleted_count} existing documents")
        
        # Insert sample data
        result = collection.insert_many(sample_planets)
        print(f"✅ Inserted {len(result.inserted_ids)} planet documents")
        
        # Create index on kepid for faster queries
        collection.create_index("kepid", unique=True)
        print(f"✅ Created index on kepid field")
        
        # Display inserted data
        print(f"\n📊 Sample Data Summary:")
        for planet in sample_planets:
            print(f"   - KEPID {planet['kepid']}: {planet['planet_type']} "
                  f"({planet['radius_earth']}x Earth radius)")
        
        return True
    except Exception as e:
        print(f"❌ Error populating data: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("MongoDB Population Script for Exoplanet Data")
    print("=" * 60)
    
    # Connect to MongoDB
    collection = connect_to_mongodb()
    if not collection:
        print("\n❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Populate data
    print("\n📝 Populating sample planet data...")
    success = populate_sample_data(collection)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ MongoDB population completed successfully!")
        print("=" * 60)
        print("\n💡 Tips:")
        print("   - Use GET /planets endpoint to see enriched data")
        print("   - MongoDB data will be merged with CSV data")
        print("   - MongoDB fields take precedence over CSV fields")
        print("   - You can add more planets using MongoDB shell or this script")
    else:
        print("\n❌ MongoDB population failed!")

if __name__ == "__main__":
    main()