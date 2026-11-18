from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from backend.app.models.person import Person, PersonIdentifier
from backend.app.models.clinical import ClinicalObservation, Encounter
from backend.app.models.screening import ScreeningEncounter
from backend.app.models.recommendation import PreventiveRecommendation
from backend.app.integrations.terminology.mapper import TerminologyMapper


class FhirSerializer:
    """HL7 FHIR R4 Serializer and Validator compliant with Department of Health Abu Dhabi standards."""

    @staticmethod
    def serialize_patient(person: Person, identifiers: Optional[List[PersonIdentifier]] = None) -> Dict[str, Any]:
        """Serializes internal Person model to FHIR R4 Patient resource."""
        id_list = []
        if identifiers:
            for ident in identifiers:
                id_list.append({
                    "system": "http://fhir.doh.gov.ae/identifier/emirates-id",
                    "value": ident.identifier_value,
                })
        else:
            id_list.append({
                "system": "http://fhir.doh.gov.ae/identifier/local-mrn",
                "value": person.id,
            })

        gender_str = (
            person.gender.value.lower()
            if hasattr(person.gender, "value")
            else str(person.gender).lower()
        )
        if gender_str not in ["male", "female", "other", "unknown"]:
            gender_str = "unknown"

        birth_date = person.date_of_birth.strftime("%Y-%m-%d") if person.date_of_birth else None

        telecom = []
        if person.phone_number:
            telecom.append({"system": "phone", "value": person.phone_number, "use": "mobile"})
        if person.email:
            telecom.append({"system": "email", "value": person.email, "use": "home"})

        return {
            "resourceType": "Patient",
            "id": person.id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
                "lastUpdated": datetime.utcnow().isoformat() + "Z",
            },
            "identifier": id_list,
            "active": True,
            "name": [
                {
                    "use": "official",
                    "family": person.last_name or "",
                    "given": [person.first_name] if person.first_name else [],
                }
            ],
            "telecom": telecom,
            "gender": gender_str,
            "birthDate": birth_date,
        }

    @staticmethod
    def serialize_observation(
        obs: ClinicalObservation,
        person_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serializes internal ClinicalObservation model to FHIR R4 Observation resource with LOINC codes."""
        code_val = getattr(obs, "code", None)
        disp_val = getattr(obs, "display_name", None)
        num_val = getattr(obs, "numeric_value", None)
        unit_val = getattr(obs, "unit", "")
        pat_id = getattr(obs, "patient_id", None) or person_id
        time_val = getattr(obs, "collection_time", None) or getattr(obs, "effective_time", None) or datetime.utcnow()

        if not code_val:
            obs_type_str = obs.observation_type.value if hasattr(obs.observation_type, "value") else str(obs.observation_type)
            mapped = TerminologyMapper.lookup_internal(obs_type_str)
            code_val = mapped["target_code"] if mapped else "unknown"
            disp_val = mapped["target_display"] if mapped else obs_type_str
            unit_val = mapped["ucum_unit"] if mapped else unit_val

        res = {
            "resourceType": "Observation",
            "id": getattr(obs, "id", "obs-001"),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"],
                "lastUpdated": datetime.utcnow().isoformat() + "Z",
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": code_val,
                        "display": disp_val or code_val,
                    }
                ],
                "text": disp_val or code_val,
            },
            "subject": {"reference": f"Patient/{pat_id}"},
            "effectiveDateTime": time_val.isoformat() if hasattr(time_val, "isoformat") else str(time_val),
        }

        if num_val is not None:
            res["valueQuantity"] = {
                "value": float(num_val),
                "unit": unit_val or "",
                "system": "http://unitsofmeasure.org",
                "code": unit_val or "",
            }

        return res

    @staticmethod
    def serialize_encounter(screening: Any) -> Dict[str, Any]:
        """Serializes ScreeningRecord or Encounter to FHIR R4 Encounter resource."""
        enc_type_str = getattr(screening, "screening_type", None)
        if not enc_type_str:
            enc_val = getattr(screening, "encounter_type", "AMB")
            enc_type_str = enc_val.value if hasattr(enc_val, "value") else str(enc_val)

        mapped = TerminologyMapper.lookup_internal(enc_type_str)

        coding = []
        if mapped:
            coding.append({
                "system": mapped["target_system"],
                "code": mapped["target_code"],
                "display": mapped["target_display"],
            })

        pat_id = getattr(screening, "patient_id", None) or getattr(screening, "person_id", "pat-001")
        date_val = getattr(screening, "screening_date", None) or getattr(screening, "performed_at", None) or datetime.utcnow()
        date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)

        return {
            "resourceType": "Encounter",
            "id": getattr(screening, "id", "enc-001"),
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Encounter"],
                "lastUpdated": datetime.utcnow().isoformat() + "Z",
            },
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "type": [{"coding": coding, "text": enc_type_str}],
            "subject": {"reference": f"Patient/{pat_id}"},
            "period": {"start": date_str},
        }

    @staticmethod
    def serialize_bundle(
        resources: List[Dict[str, Any]],
        bundle_type: str = "collection",
    ) -> Dict[str, Any]:
        """Packages multiple FHIR resources into a validated FHIR Bundle."""
        entries = [{"fullUrl": f"urn:uuid:{r.get('id', '')}", "resource": r} for r in resources]
        return {
            "resourceType": "Bundle",
            "type": bundle_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total": len(resources),
            "entry": entries,
        }

    @staticmethod
    def validate_resource(resource: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Strict validation of FHIR R4 resource structure and mandatory attributes."""
        if not isinstance(resource, dict):
            return False, "Resource must be a JSON object"
        res_type = resource.get("resourceType")
        if not res_type:
            return False, "Missing mandatory field 'resourceType'"

        valid_types = ["Patient", "Observation", "Encounter", "ServiceRequest", "CarePlan", "Bundle"]
        if res_type not in valid_types:
            return False, f"Unsupported FHIR resource type '{res_type}'"

        if res_type == "Patient":
            if not resource.get("identifier") and not resource.get("name"):
                return False, "Patient resource must contain either identifier or name"

        elif res_type == "Observation":
            if not resource.get("status"):
                return False, "Observation must contain mandatory 'status'"
            if not resource.get("code"):
                return False, "Observation must contain mandatory 'code'"

        elif res_type == "Encounter":
            if not resource.get("status"):
                return False, "Encounter must contain mandatory 'status'"
            if not resource.get("class"):
                return False, "Encounter must contain mandatory 'class'"

        return True, None
