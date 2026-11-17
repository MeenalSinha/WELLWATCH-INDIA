"""Recommendations, Cultural Outreach & Care Navigation Schema

Revision ID: 005_recommendations_outreach_navigation
Revises: 004_genomic_intelligence
Create Date: 2026-11-17 17:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '005_recommendations_outreach_navigation'
down_revision: Union[str, None] = '004_genomic_intelligence'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. preventive_recommendations
    op.create_table(
        'preventive_recommendations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('assessment_id', sa.String(length=36), sa.ForeignKey('preventive_assessments.id'), nullable=True),
        sa.Column('pathway_id', sa.String(length=100), nullable=False),
        sa.Column('care_gap_id', sa.String(length=100), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('lifecycle_status', sa.String(length=50), nullable=False),
        sa.Column('recommended_action', sa.String(length=255), nullable=False),
        sa.Column('clinical_rationale', sa.Text(), nullable=False),
        sa.Column('supporting_evidence', sa.JSON(), nullable=False),
        sa.Column('source_guideline', sa.String(length=255), nullable=False),
        sa.Column('uncertainty_status', sa.String(length=50), nullable=False),
        sa.Column('rule_id', sa.String(length=100), nullable=False),
        sa.Column('rule_version', sa.String(length=50), nullable=False),
        sa.Column('target_date', sa.DateTime(), nullable=True),
        sa.Column('expiration_date', sa.DateTime(), nullable=True),
        sa.Column('requires_human_review', sa.Boolean(), nullable=False, default=True),
        sa.Column('declined_reason', sa.String(length=255), nullable=True),
        sa.Column('assigned_provider_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('reviewer_notes', sa.Text(), nullable=True),
    )
    op.create_index('ix_rec_person_status', 'preventive_recommendations', ['person_id', 'lifecycle_status'])
    op.create_index('ix_rec_org_priority', 'preventive_recommendations', ['organization_id', 'priority'])

    # 2. recommendation_reviews
    op.create_table(
        'recommendation_reviews',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=False),
        sa.Column('reviewer_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('review_decision', sa.String(length=50), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('modifications', sa.JSON(), nullable=True),
    )

    # 3. preventive_care_plans
    op.create_table(
        'preventive_care_plans',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('plan_name', sa.String(length=255), nullable=False),
        sa.Column('pathway_id', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('goals', sa.JSON(), nullable=False),
        sa.Column('responsible_org_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('assigned_care_team', sa.JSON(), nullable=False),
        sa.Column('target_date', sa.DateTime(), nullable=True),
        sa.Column('review_schedule', sa.String(length=100), nullable=True),
    )

    # 4. care_plan_items
    op.create_table(
        'care_plan_items',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('care_plan_id', sa.String(length=36), sa.ForeignKey('preventive_care_plans.id'), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('target_date', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )

    # 5. communication_preferences
    op.create_table(
        'communication_preferences',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False, unique=True),
        sa.Column('preferred_language', sa.String(length=10), nullable=False),
        sa.Column('preferred_channel', sa.String(length=20), nullable=False),
        sa.Column('contact_window_start', sa.String(length=10), nullable=False),
        sa.Column('contact_window_end', sa.String(length=10), nullable=False),
        sa.Column('quiet_hours_start', sa.String(length=10), nullable=False),
        sa.Column('quiet_hours_end', sa.String(length=10), nullable=False),
        sa.Column('consent_opt_in', sa.Boolean(), nullable=False),
        sa.Column('opt_out_timestamp', sa.DateTime(), nullable=True),
        sa.Column('guardian_proxy_required', sa.Boolean(), nullable=False),
        sa.Column('accessibility_needs', sa.JSON(), nullable=True),
    )

    # 6. outreach_tasks
    op.create_table(
        'outreach_tasks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=True),
        sa.Column('recipient_id', sa.String(length=36), nullable=True),
        sa.Column('recipient_role', sa.String(length=50), nullable=False),
        sa.Column('channel', sa.String(length=20), nullable=False),
        sa.Column('template_id', sa.String(length=100), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=False),
        sa.Column('message_body', sa.Text(), nullable=False),
        sa.Column('scheduled_at', sa.DateTime(), nullable=False),
        sa.Column('attempted_at', sa.DateTime(), nullable=True),
        sa.Column('delivery_status', sa.String(length=50), nullable=False),
        sa.Column('provider_status', sa.String(length=100), nullable=False),
        sa.Column('provider_message_id', sa.String(length=100), nullable=True),
        sa.Column('failure_reason', sa.String(length=255), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('max_retries', sa.Integer(), nullable=False, default=3),
        sa.Column('consent_verified', sa.Boolean(), nullable=False, default=True),
    )
    op.create_index('ix_outreach_person_status', 'outreach_tasks', ['person_id', 'delivery_status'])
    op.create_index('ix_outreach_scheduled', 'outreach_tasks', ['scheduled_at', 'delivery_status'])

    # 7. appointments
    op.create_table(
        'appointments',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=True),
        sa.Column('care_plan_id', sa.String(length=36), sa.ForeignKey('preventive_care_plans.id'), nullable=True),
        sa.Column('service_type', sa.String(length=100), nullable=False),
        sa.Column('scheduled_time', sa.DateTime(), nullable=False),
        sa.Column('location', sa.String(length=255), nullable=False),
        sa.Column('provider_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('attendance_notes', sa.Text(), nullable=True),
    )

    # 8. referrals
    op.create_table(
        'referrals',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=True),
        sa.Column('referring_provider_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('receiving_org_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=True),
        sa.Column('specialty', sa.String(length=100), nullable=False),
        sa.Column('clinical_indication', sa.Text(), nullable=False),
        sa.Column('priority', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('outcome_notes', sa.Text(), nullable=True),
    )

    # 9. follow_up_tasks
    op.create_table(
        'follow_up_tasks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=True),
        sa.Column('appointment_id', sa.String(length=36), sa.ForeignKey('appointments.id'), nullable=True),
        sa.Column('referral_id', sa.String(length=36), sa.ForeignKey('referrals.id'), nullable=True),
        sa.Column('trigger_reason', sa.String(length=100), nullable=False),
        sa.Column('assigned_role', sa.String(length=50), nullable=False),
        sa.Column('assigned_user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('due_date', sa.DateTime(), nullable=False),
        sa.Column('priority', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('resolution_evidence_id', sa.String(length=100), nullable=True),
    )

    # 10. recall_tasks
    op.create_table(
        'recall_tasks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('pathway_id', sa.String(length=100), nullable=False),
        sa.Column('screening_type', sa.String(length=100), nullable=False),
        sa.Column('due_date', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('reminder_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_reminded_at', sa.DateTime(), nullable=True),
    )

    # 11. care_gap_closure_evidence
    op.create_table(
        'care_gap_closure_evidence',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('care_gap_id', sa.String(length=100), nullable=False),
        sa.Column('recommendation_id', sa.String(length=36), sa.ForeignKey('preventive_recommendations.id'), nullable=True),
        sa.Column('evidence_source', sa.String(length=100), nullable=False),
        sa.Column('evidence_id', sa.String(length=100), nullable=False),
        sa.Column('verification_method', sa.String(length=100), nullable=False),
        sa.Column('verifying_user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('verification_notes', sa.Text(), nullable=True),
        sa.Column('verified_at', sa.DateTime(), nullable=False),
        sa.Column('is_valid', sa.Boolean(), nullable=False, default=True),
    )


def downgrade() -> None:
    op.drop_table('care_gap_closure_evidence')
    op.drop_table('recall_tasks')
    op.drop_table('follow_up_tasks')
    op.drop_table('referrals')
    op.drop_table('appointments')
    op.drop_table('outreach_tasks')
    op.drop_table('communication_preferences')
    op.drop_table('care_plan_items')
    op.drop_table('preventive_care_plans')
    op.drop_table('recommendation_reviews')
    op.drop_table('preventive_recommendations')
