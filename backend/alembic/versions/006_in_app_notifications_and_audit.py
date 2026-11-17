"""In-app notifications and outreach authorization status

Revision ID: 006_in_app_notifications_and_audit
Revises: 005_recommendations_outreach_navigation
Create Date: 2026-11-18 03:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '006_in_app_notifications_and_audit'
down_revision: Union[str, None] = '005_recommendations_outreach_navigation'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add authorization_status column to communication_preferences
    with op.batch_alter_table('communication_preferences') as batch_op:
        batch_op.add_column(
            sa.Column('authorization_status', sa.String(length=50), nullable=False, server_default='NOT_PROVIDED')
        )

    # 2. in_app_notifications
    op.create_table(
        'in_app_notifications',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('person_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('recipient_user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('outreach_task_id', sa.String(length=36), sa.ForeignKey('outreach_tasks.id'), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('message_body', sa.Text(), nullable=False),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('read_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_inapp_person', 'in_app_notifications', ['person_id'])
    op.create_index('ix_inapp_recipient', 'in_app_notifications', ['recipient_user_id'])
    op.create_index('ix_inapp_read', 'in_app_notifications', ['is_read'])


def downgrade() -> None:
    op.drop_table('in_app_notifications')
    with op.batch_alter_table('communication_preferences') as batch_op:
        batch_op.drop_column('authorization_status')
