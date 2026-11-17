"""Preventive Assessments Schema

Revision ID: 003_preventive_assessments
Revises: 002_longitudinal_data_model
Create Date: 2026-11-17 14:40:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003_preventive_assessments'
down_revision: Union[str, None] = '002_longitudinal_data_model'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'preventive_assessments',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('assessment_type', sa.String(length=100), nullable=False, server_default='COMPREHENSIVE_PREVENTIVE_ASSESSMENT'),
        sa.Column('population_category', sa.String(length=50), nullable=False),
        sa.Column('pathway_id', sa.String(length=100), nullable=False),
        sa.Column('pathway_version', sa.String(length=50), nullable=False),
        sa.Column('assessment_status', sa.String(length=50), nullable=False, server_default='COMPLETED'),
        sa.Column('clinical_risk_tier', sa.String(length=50), nullable=False, server_default='INSUFFICIENT_DATA'),
        sa.Column('eligibility_summary', sa.Text(), nullable=False),
        sa.Column('care_gaps_summary', sa.Text(), nullable=False),
        sa.Column('contributing_factors', sa.Text(), nullable=False),
        sa.Column('data_quality_limitations', sa.Text(), nullable=False),
        sa.Column('human_review_status', sa.String(length=50), nullable=False, server_default='AWAITING_REVIEW'),
        sa.Column('reviewed_by_user_id', sa.String(length=36), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewer_notes', sa.Text(), nullable=True),
        sa.Column('recommended_action', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reviewed_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_assessments_patient', 'preventive_assessments', ['patient_id'])
    op.create_index('ix_assessments_status', 'preventive_assessments', ['assessment_status'])
    op.create_index('ix_assessments_risk_tier', 'preventive_assessments', ['clinical_risk_tier'])
    op.create_index('ix_assessments_review_status', 'preventive_assessments', ['human_review_status'])
    op.create_index('ix_assessments_patient_status', 'preventive_assessments', ['patient_id', 'assessment_status'])


def downgrade() -> None:
    op.drop_table('preventive_assessments')
