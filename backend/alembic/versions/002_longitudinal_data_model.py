"""Longitudinal Healthcare Data Model

Revision ID: 002_longitudinal_data_model
Revises: 001_initial_schema
Create Date: 2026-11-17 14:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002_longitudinal_data_model'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Persons Table
    op.create_table(
        'persons',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('first_name', sa.String(length=100), nullable=False),
        sa.Column('last_name', sa.String(length=100), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=False),
        sa.Column('gender', sa.String(length=20), nullable=False),
        sa.Column('preferred_language', sa.String(length=10), nullable=False, server_default='ar'),
        sa.Column('nationality', sa.String(length=50), nullable=True, server_default='ARE'),
        sa.Column('emirate', sa.String(length=50), nullable=True, server_default='Abu Dhabi'),
        sa.Column('district', sa.String(length=100), nullable=True),
        sa.Column('record_status', sa.String(length=20), nullable=False, server_default='ACTIVE'),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index('ix_persons_dob', 'persons', ['date_of_birth'])
    op.create_index('ix_persons_status', 'persons', ['record_status'])
    op.create_index('ix_persons_dob_gender', 'persons', ['date_of_birth', 'gender'])

    # 2. Person Identifiers Table
    op.create_table(
        'person_identifiers',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('person_id', sa.String(length=36), nullable=False),
        sa.Column('namespace', sa.String(length=50), nullable=False),
        sa.Column('issuing_authority', sa.String(length=100), nullable=False, server_default='ICP UAE'),
        sa.Column('identifier_value', sa.String(length=255), nullable=False),
        sa.Column('identifier_hash', sa.String(length=64), nullable=True),
        sa.Column('verification_status', sa.String(length=30), nullable=False, server_default='UNVERIFIED'),
        sa.Column('valid_from', sa.Date(), nullable=True),
        sa.Column('valid_until', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('namespace', 'identifier_value', name='uq_identifier_namespace_val')
    )
    op.create_index('ix_identifiers_person', 'person_identifiers', ['person_id'])
    op.create_index('ix_identifiers_lookup', 'person_identifiers', ['namespace', 'identifier_hash'])

    # 3. Patient Profiles Table
    op.create_table(
        'patient_profiles',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('person_id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('population_classification', sa.String(length=30), nullable=False),
        sa.Column('completeness_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('completeness_status', sa.String(length=30), nullable=False, server_default='MINIMAL'),
        sa.Column('last_assessment_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('primary_facility_name', sa.String(length=200), nullable=True),
        sa.Column('is_high_risk', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('person_id')
    )
    op.create_index('ix_patient_profiles_org', 'patient_profiles', ['organization_id'])
    op.create_index('ix_patient_profiles_pop', 'patient_profiles', ['population_classification'])

    # 4. Households Table
    op.create_table(
        'households',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('household_name', sa.String(length=150), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('primary_contact_person_id', sa.String(length=36), nullable=True),
        sa.Column('emirate', sa.String(length=50), nullable=False, server_default='Abu Dhabi'),
        sa.Column('district', sa.String(length=100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['primary_contact_person_id'], ['persons.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_households_org', 'households', ['organization_id'])

    # 5. Household Members Table
    op.create_table(
        'household_members',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('household_id', sa.String(length=36), nullable=False),
        sa.Column('person_id', sa.String(length=36), nullable=False),
        sa.Column('relationship_to_head', sa.String(length=50), nullable=False, server_default='MEMBER'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['household_id'], ['households.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('household_id', 'person_id', name='uq_household_person')
    )

    # 6. Data Sources Table
    op.create_table(
        'data_sources',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=True),
        sa.Column('system_name', sa.String(length=150), nullable=False),
        sa.Column('system_version', sa.String(length=50), nullable=False, server_default='1.0'),
        sa.Column('trust_level', sa.String(length=50), nullable=False, server_default='ACCREDITED_CLINICAL'),
        sa.Column('connection_status', sa.String(length=50), nullable=False, server_default='NOT_CONFIGURED'),
        sa.Column('last_sync_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('endpoint_contract_ref', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_data_sources_type', 'data_sources', ['source_type'])

    # 7. Data Provenances Table
    op.create_table(
        'data_provenances',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('data_source_id', sa.String(length=36), nullable=False),
        sa.Column('collection_method', sa.String(length=100), nullable=False, server_default='DIRECT_CLINICAL_ENTRY'),
        sa.Column('import_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('external_record_id', sa.String(length=150), nullable=True),
        sa.Column('verification_status', sa.String(length=30), nullable=False, server_default='UNVERIFIED'),
        sa.Column('transformation_history', sa.Text(), nullable=True),
        sa.Column('responsible_processor_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['responsible_processor_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_provenance_source', 'data_provenances', ['data_source_id'])

    # 8. Encounters Table
    op.create_table(
        'encounters',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('facility_id', sa.String(length=36), nullable=False),
        sa.Column('provider_user_id', sa.String(length=36), nullable=True),
        sa.Column('encounter_type', sa.String(length=50), nullable=False),
        sa.Column('encounter_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False, server_default='COMPLETED'),
        sa.Column('clinical_notes', sa.Text(), nullable=True),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['facility_id'], ['organizations.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['provider_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_encounters_patient', 'encounters', ['patient_id'])
    op.create_index('ix_encounters_facility', 'encounters', ['facility_id'])
    op.create_index('ix_encounters_date', 'encounters', ['encounter_date'])

    # 9. Clinical Observations Table
    op.create_table(
        'clinical_observations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('encounter_id', sa.String(length=36), nullable=True),
        sa.Column('observation_type', sa.String(length=50), nullable=False),
        sa.Column('code_system', sa.String(length=50), nullable=False, server_default='LOINC'),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('display_name', sa.String(length=150), nullable=False),
        sa.Column('numeric_value', sa.Float(), nullable=True),
        sa.Column('string_value', sa.String(length=255), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('canonical_value', sa.Float(), nullable=True),
        sa.Column('canonical_unit', sa.String(length=50), nullable=True),
        sa.Column('collection_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('effective_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('recording_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('epistemic_status', sa.String(length=30), nullable=False, server_default='VERIFIED'),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['encounter_id'], ['encounters.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_obs_patient', 'clinical_observations', ['patient_id'])
    op.create_index('ix_obs_code', 'clinical_observations', ['code'])
    op.create_index('ix_obs_collection_time', 'clinical_observations', ['collection_time'])
    op.create_index('ix_obs_epistemic', 'clinical_observations', ['epistemic_status'])
    op.create_index('ix_obs_patient_code_date', 'clinical_observations', ['patient_id', 'code', 'collection_time'])

    # 10. Clinical Conditions Table
    op.create_table(
        'clinical_conditions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('code_system', sa.String(length=50), nullable=False, server_default='ICD-10'),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('clinical_description', sa.String(length=255), nullable=False),
        sa.Column('clinical_status', sa.String(length=30), nullable=False, server_default='ACTIVE'),
        sa.Column('verification_status', sa.String(length=30), nullable=False, server_default='CONFIRMED'),
        sa.Column('onset_date', sa.Date(), nullable=True),
        sa.Column('resolution_date', sa.Date(), nullable=True),
        sa.Column('recording_source_id', sa.String(length=36), nullable=True),
        sa.Column('diagnosing_provider_id', sa.String(length=36), nullable=True),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['recording_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['diagnosing_provider_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conditions_patient', 'clinical_conditions', ['patient_id'])
    op.create_index('ix_conditions_code', 'clinical_conditions', ['code'])

    # 11. Family Histories Table
    op.create_table(
        'family_histories',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('relationship_to_patient', sa.String(length=50), nullable=False),
        sa.Column('condition_code', sa.String(length=50), nullable=True),
        sa.Column('condition_name', sa.String(length=200), nullable=False),
        sa.Column('approximate_age_of_onset', sa.Integer(), nullable=True),
        sa.Column('is_deceased', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('verification_status', sa.String(length=30), nullable=False, server_default='UNVERIFIED'),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_fam_hist_patient', 'family_histories', ['patient_id'])

    # 12. Screening Records Table
    op.create_table(
        'screening_records',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('encounter_id', sa.String(length=36), nullable=True),
        sa.Column('screening_type', sa.String(length=100), nullable=False),
        sa.Column('screening_date', sa.Date(), nullable=False),
        sa.Column('screening_status', sa.String(length=30), nullable=False, server_default='COMPLETED'),
        sa.Column('completion_evidence_ref', sa.String(length=255), nullable=True),
        sa.Column('performing_facility_id', sa.String(length=36), nullable=True),
        sa.Column('findings_summary', sa.Text(), nullable=True),
        sa.Column('epistemic_status', sa.String(length=30), nullable=False, server_default='VERIFIED'),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['encounter_id'], ['encounters.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['performing_facility_id'], ['organizations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_screenings_patient', 'screening_records', ['patient_id'])
    op.create_index('ix_screenings_type', 'screening_records', ['screening_type'])
    op.create_index('ix_screenings_date', 'screening_records', ['screening_date'])
    op.create_index('ix_screenings_patient_type_date', 'screening_records', ['patient_id', 'screening_type', 'screening_date'])

    # 13. Clinical Documents Table
    op.create_table(
        'clinical_documents',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('encounter_id', sa.String(length=36), nullable=True),
        sa.Column('document_type', sa.String(length=100), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False, server_default='application/pdf'),
        sa.Column('storage_uri', sa.String(length=500), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['encounter_id'], ['encounters.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_documents_patient', 'clinical_documents', ['patient_id'])

    # 14. Add Phase 4 columns to Guardianships
    with op.batch_alter_table('guardianships') as batch_op:
        batch_op.add_column(sa.Column('guardian_person_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('dependent_person_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('authorization_scope', sa.String(length=100), nullable=False, server_default='FULL_PREVENTIVE_HEALTHCARE_PROXY'))
        batch_op.add_column(sa.Column('valid_from', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('revocation_reason', sa.Text(), nullable=True))
        batch_op.create_foreign_key('fk_guardianship_guardian_person', 'persons', ['guardian_person_id'], ['id'], ondelete='CASCADE')
        batch_op.create_foreign_key('fk_guardianship_dependent_person', 'persons', ['dependent_person_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    with op.batch_alter_table('guardianships') as batch_op:
        batch_op.drop_constraint('fk_guardianship_guardian_person', type_='foreignkey')
        batch_op.drop_constraint('fk_guardianship_dependent_person', type_='foreignkey')
        batch_op.drop_column('revocation_reason')
        batch_op.drop_column('revoked_at')
        batch_op.drop_column('valid_until')
        batch_op.drop_column('valid_from')
        batch_op.drop_column('authorization_scope')
        batch_op.drop_column('dependent_person_id')
        batch_op.drop_column('guardian_person_id')

    op.drop_table('clinical_documents')
    op.drop_table('screening_records')
    op.drop_table('family_histories')
    op.drop_table('clinical_conditions')
    op.drop_table('clinical_observations')
    op.drop_table('encounters')
    op.drop_table('data_provenances')
    op.drop_table('data_sources')
    op.drop_table('household_members')
    op.drop_table('households')
    op.drop_table('patient_profiles')
    op.drop_table('person_identifiers')
    op.drop_table('persons')
