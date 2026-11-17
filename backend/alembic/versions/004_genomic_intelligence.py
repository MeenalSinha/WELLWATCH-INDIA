"""Genomic Intelligence Schema

Revision ID: 004_genomic_intelligence
Revises: 003_preventive_assessments
Create Date: 2026-11-17 15:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '004_genomic_intelligence'
down_revision: Union[str, None] = '003_preventive_assessments'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'genomic_findings',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), nullable=False),
        sa.Column('gene_symbol', sa.String(length=64), nullable=False),
        sa.Column('variant_id', sa.String(length=128), nullable=False),
        sa.Column('genomic_coordinates', sa.String(length=128), nullable=True),
        sa.Column('variant_classification', sa.String(length=50), nullable=False),
        sa.Column('clinical_interpretation', sa.Text(), nullable=False),
        sa.Column('actionable_guidance', sa.Text(), nullable=True),
        sa.Column('source_laboratory', sa.String(length=255), nullable=False),
        sa.Column('report_reference', sa.String(length=128), nullable=True),
        sa.Column('provenance_id', sa.String(length=36), nullable=True),
        sa.Column('review_status', sa.String(length=50), nullable=False, server_default='PENDING_REVIEW'),
        sa.Column('reviewed_by_user_id', sa.String(length=36), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewer_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['persons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['provenance_id'], ['data_provenances.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reviewed_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_genomic_findings_patient', 'genomic_findings', ['patient_id'])
    op.create_index('ix_genomic_findings_gene', 'genomic_findings', ['gene_symbol'])
    op.create_index('ix_genomic_findings_patient_gene', 'genomic_findings', ['patient_id', 'gene_symbol'])


def downgrade() -> None:
    op.drop_index('ix_genomic_findings_patient_gene', table_name='genomic_findings')
    op.drop_index('ix_genomic_findings_gene', table_name='genomic_findings')
    op.drop_index('ix_genomic_findings_patient', table_name='genomic_findings')
    op.drop_table('genomic_findings')
