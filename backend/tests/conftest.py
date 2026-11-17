import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.core.database import Base, get_db
from backend.app.core.security import get_password_hash
from backend.app.main import app
from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole

# File-based isolated test DB
TEST_DB_PATH = "./data/test_wellwatch.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

engine_test = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

# Pre-compute password hash once to ensure blistering fast test suite execution
PRECOMPUTED_HASH = get_password_hash("ValidPass123!")


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_organization(db_session) -> Organization:
    existing = db_session.query(Organization).filter(Organization.license_number == "DOH-ORG-00123").first()
    if existing:
        return existing
        
    org = Organization(
        name="SEHA Al Bateen Healthcare Center",
        license_number="DOH-ORG-00123",
        organization_type="PRIMARY_CARE_CLINIC",
        region="ABU_DHABI_CITY",
        is_active=True
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture
def test_organization_2(db_session) -> Organization:
    existing = db_session.query(Organization).filter(Organization.license_number == "DOH-ORG-00456").first()
    if existing:
        return existing

    org = Organization(
        name="SEHA Al Ain Regional Hospital",
        license_number="DOH-ORG-00456",
        organization_type="HOSPITAL",
        region="AL_AIN",
        is_active=True
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture
def test_users(db_session, test_organization) -> dict:
    """Pre-provisions verified test users for all 8 roles."""
    users = {}
    roles = [
        UserRole.INDIVIDUAL,
        UserRole.PARENT_GUARDIAN,
        UserRole.COMMUNITY_HEALTH_WORKER,
        UserRole.HEALTHCARE_PROVIDER,
        UserRole.CARE_COORDINATOR,
        UserRole.POPULATION_HEALTH_ADMIN,
        UserRole.SYSTEM_ADMIN,
        UserRole.DATA_GOVERNANCE_OFFICER,
    ]

    for role in roles:
        email = f"{role.value.lower()}@wellwatch.ae"
        user = db_session.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                full_name=f"Test {role.value.replace('_', ' ').title()}",
                password_hash=PRECOMPUTED_HASH,
                role=role.value,
                organization_id=test_organization.id if role not in [UserRole.INDIVIDUAL, UserRole.PARENT_GUARDIAN] else None,
                is_active=True,
                is_verified=True
            )
            db_session.add(user)
            db_session.commit()
            db_session.refresh(user)
        else:
            # Reset lockout / attempts if any
            user.failed_login_count = 0
            user.locked_until = None
            db_session.commit()
            db_session.refresh(user)
        users[role] = user

    # Child user
    child_email = "child.dependent@wellwatch.ae"
    child_user = db_session.query(User).filter(User.email == child_email).first()
    if not child_user:
        child_user = User(
            email=child_email,
            full_name="Baby Zayed",
            password_hash=PRECOMPUTED_HASH,
            role=UserRole.INDIVIDUAL.value,
            is_active=True,
            is_verified=True
        )
        db_session.add(child_user)
        db_session.commit()
        db_session.refresh(child_user)
    users["CHILD"] = child_user

    return users
