"""Phases 12 to 14 schema: Population Health, Offline Sync, and Healthcare Interoperability

Revision ID: 007_phases_12_to_14_schema
Revises: 006_in_app_notifications_and_audit
Create Date: 2026-11-18 04:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '007_phases_12_to_14_schema'
down_revision: Union[str, None] = '006_in_app_notifications_and_audit'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- PHASE 12: POPULATION HEALTH ---
    op.create_table(
        'population_metric_snapshots',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=True),
        sa.Column('snapshot_date', sa.DateTime(), nullable=False),
        sa.Column('metric_category', sa.String(length=50), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('dimension_name', sa.String(length=50), nullable=True),
        sa.Column('dimension_value', sa.String(length=100), nullable=True),
        sa.Column('numerator', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('denominator', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('metric_value', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('is_suppressed', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('metadata_json', sa.Text(), nullable=True),
    )
    op.create_index('ix_pop_snapshot_org', 'population_metric_snapshots', ['organization_id'])
    op.create_index('ix_pop_snapshot_date', 'population_metric_snapshots', ['snapshot_date'])
    op.create_index('ix_pop_snapshot_category', 'population_metric_snapshots', ['metric_category'])

    op.create_table(
        'analytics_access_logs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=True),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column('query_params', sa.Text(), nullable=False, server_default='{}'),
        sa.Column('cell_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('was_suppressed', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('accessed_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_analytics_user', 'analytics_access_logs', ['user_id'])
    op.create_index('ix_analytics_date', 'analytics_access_logs', ['accessed_at'])

    # --- PHASE 13: OFFLINE-FIRST CHW ---
    op.create_table(
        'device_registrations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('device_id', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('device_model', sa.String(length=100), nullable=False, server_default='Generic Tablet/Mobile'),
        sa.Column('os_platform', sa.String(length=50), nullable=False, server_default='WebPWA'),
        sa.Column('app_version', sa.String(length=50), nullable=False, server_default='1.0.0'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='ACTIVE'),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('registered_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_device_id', 'device_registrations', ['device_id'])
    op.create_index('ix_device_user', 'device_registrations', ['user_id'])

    op.create_table(
        'offline_assignments',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('patient_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('assignment_reason', sa.String(length=255), nullable=False, server_default='FIELD_PREVENTIVE_VISIT'),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='ACTIVE'),
    )
    op.create_index('ix_offline_assign_user', 'offline_assignments', ['user_id'])
    op.create_index('ix_offline_assign_patient', 'offline_assignments', ['patient_id'])
    op.create_index('ix_offline_assign_expiry', 'offline_assignments', ['expires_at'])

    op.create_table(
        'sync_operation_audits',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('operation_id', sa.String(length=64), unique=True, nullable=False),
        sa.Column('idempotency_key', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('device_id', sa.String(length=128), nullable=True),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=36), nullable=True),
        sa.Column('operation_type', sa.String(length=20), nullable=False, server_default='CREATE'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='SYNCED'),
        sa.Column('payload_snapshot', sa.Text(), nullable=False),
        sa.Column('client_timestamp', sa.DateTime(), nullable=False),
        sa.Column('server_timestamp', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_sync_op_idempotency', 'sync_operation_audits', ['idempotency_key'])
    op.create_index('ix_sync_op_user', 'sync_operation_audits', ['user_id'])

    op.create_table(
        'sync_conflicts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('operation_id', sa.String(length=64), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.String(length=36), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('conflict_reason', sa.String(length=255), nullable=False),
        sa.Column('client_payload', sa.Text(), nullable=False),
        sa.Column('server_payload', sa.Text(), nullable=False),
        sa.Column('resolution_status', sa.String(length=30), nullable=False, server_default='PENDING_REVIEW'),
        sa.Column('resolved_by', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_conflict_patient', 'sync_conflicts', ['patient_id'])
    op.create_index('ix_conflict_status', 'sync_conflicts', ['resolution_status'])

    # --- PHASE 14: HEALTHCARE INTEROPERABILITY ---
    op.create_table(
        'integration_providers',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('name', sa.String(length=50), unique=True, nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('provider_type', sa.String(length=20), nullable=False),
        sa.Column('endpoint_url', sa.String(length=255), nullable=True),
        sa.Column('auth_type', sa.String(length=20), nullable=False, server_default='OAUTH2'),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='NOT_CONFIGURED'),
        sa.Column('configuration_json', sa.Text(), nullable=True),
        sa.Column('last_heartbeat_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_integration_prov_type', 'integration_providers', ['provider_type'])

    op.create_table(
        'integration_job_logs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('provider_id', sa.String(length=36), sa.ForeignKey('integration_providers.id'), nullable=False),
        sa.Column('job_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='SUCCESS'),
        sa.Column('records_processed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('records_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('executed_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_integration_job_provider', 'integration_job_logs', ['provider_id'])

    op.create_table(
        'provenance_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('target_resource_type', sa.String(length=50), nullable=False),
        sa.Column('target_resource_id', sa.String(length=36), nullable=False),
        sa.Column('source_system', sa.String(length=100), nullable=False),
        sa.Column('source_identifier', sa.String(length=100), nullable=True),
        sa.Column('agent_id', sa.String(length=100), nullable=False),
        sa.Column('agent_type', sa.String(length=50), nullable=False, server_default='INTEGRATION_ADAPTER'),
        sa.Column('signature_hash', sa.String(length=128), nullable=True),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_provenance_target', 'provenance_records', ['target_resource_type', 'target_resource_id'])


def downgrade() -> None:
    op.drop_table('provenance_records')
    op.drop_table('integration_job_logs')
    op.drop_table('integration_providers')
    op.drop_table('sync_conflicts')
    op.drop_table('sync_operation_audits')
    op.drop_table('offline_assignments')
    op.drop_table('device_registrations')
    op.drop_table('analytics_access_logs')
    op.drop_table('population_metric_snapshots')
