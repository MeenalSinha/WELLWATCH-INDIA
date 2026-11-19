"""
Standalone CLI script to seed development demonstration data into WELLWATCH PREVENT.
Can be executed locally or inside Docker container:
  python -m backend.scripts.seed_demo_data
"""
import sys
from backend.app.core.database import SessionLocal
from backend.app.services.seed_service import seed_initial_development_data


def main():
    print("Beginning WELLWATCH PREVENT development database seeding...")
    with SessionLocal() as db:
        result = seed_initial_development_data(db)
        print(f"Seeding completed successfully: {result}")


if __name__ == "__main__":
    main()
