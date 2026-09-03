import streamlit as st
from planner.facility_config import ZONES, EQUIPMENT
from app.database import (
    save_field_observation, sync_queued_observations, set_equipment_status,
    save_opt_out, get_field_observations, resolve_conflict,
)


def render():
    st.title("Field Observation Capture")
    st.caption("Zone-level, aggregated observations only — no individual identity is collected "
               "(see Privacy documentation).")

    online = st.session_state.get("online", True)
    if not online:
        st.warning("Offline mode active — this observation will be queued locally and synced later.")

    with st.form("field_obs_form"):
        facility = st.text_input("Facility", value="Industrial Facility A")
        zone_id = st.selectbox("Zone", [z.zone_id for z in ZONES])
        occupancy_band = st.selectbox("Occupancy", ["LOW", "MEDIUM", "HIGH"])
        comfort_status = st.selectbox("Comfort", ["Comfortable", "Slightly warm", "Slightly cool", "Uncomfortable"])
        zone_equip = [e.equipment_id for e in EQUIPMENT if e.zone_id == zone_id] or ["(none)"]
        equipment_id = st.selectbox("Equipment", zone_equip)
        equipment_status = st.selectbox("Status", ["AVAILABLE", "MAINTENANCE"])
        can_shift = st.radio("Can load be shifted?", ["YES", "NO"], horizontal=True)
        available_reduction = st.number_input("Available reduction (kW)", min_value=0.0, value=20.0, step=5.0)
        duration = st.number_input("Duration (minutes)", min_value=0, value=30, step=15)
        notes = st.text_area("Notes", value="Normal operating condition")
        submitted = st.form_submit_button("SAVE")

    if submitted:
        obs = dict(facility=facility, zone_id=zone_id, occupancy_band=occupancy_band,
                    comfort_status=comfort_status,
                    equipment_id=None if equipment_id == "(none)" else equipment_id,
                    equipment_status=equipment_status if equipment_id != "(none)" else None,
                    can_shift_load=(can_shift == "YES"), available_reduction_kw=available_reduction,
                    duration_min=duration, notes=notes)
        result = save_field_observation(obs, online=online)
        if result["sync_state"] == "CONFLICT":
            st.error(f"CONFLICT DETECTED — {result['conflict_details']}")
        elif result["sync_state"] == "QUEUED":
            st.info("Saved to local queue (offline). Will sync when connectivity returns.")
        else:
            st.success("Observation saved and synced.")
            if equipment_id != "(none)":
                set_equipment_status(equipment_id, equipment_status)

    st.divider()
    st.subheader("Zone opt-out")
    with st.form("optout_form"):
        oo_zone = st.selectbox("Zone ", [z.zone_id for z in ZONES], key="oo_zone")
        oo_pref = st.selectbox("Preference", ["PARTICIPATE", "PARTICIPATE_ONLY_IF_NECESSARY", "TEMPORARY_OPT_OUT"])
        oo_reason = st.text_input("Reason", value="Scheduled activity")
        col1, col2 = st.columns(2)
        valid_from = col1.text_input("Valid from", value="13:00")
        valid_to = col2.text_input("Valid to", value="16:00")
        oo_submit = st.form_submit_button("Save opt-out preference")
    if oo_submit:
        save_opt_out(oo_zone, oo_pref, oo_reason, valid_from, valid_to)
        st.success(f"Preference saved for {oo_zone}: {oo_pref}")
        st.caption("This is a planning preference only — it is never used for individual "
                    "performance scoring (see Privacy documentation, Rule 3).")

    if not online:
        st.divider()
        st.subheader("Sync queue")
        if st.button("Restore connectivity & sync now"):
            result = sync_queued_observations()
            st.success(f"Synced {result['synced']} queued observation(s). "
                       f"{result['remaining_conflicts']} unresolved conflict(s) remain.")

    st.divider()
    st.subheader("Recent observations")
    obs_list = get_field_observations(limit=15)
    for o in obs_list:
        state_icon = {"SYNCED": "✅", "QUEUED": "🕒", "CONFLICT": "⚠️"}.get(o["sync_state"], "")
        st.write(f"{state_icon} `{o['created_at'][:19]}` — {o['zone_id']} — {o['sync_state']}"
                  + (f" — {o['conflict_details']}" if o.get("conflict_details") else ""))
        if o["sync_state"] == "CONFLICT":
            c1, c2 = st.columns(2)
            if c1.button("Accept local value", key=f"acc_{o['id']}"):
                resolve_conflict(o["id"], "ACCEPT_LOCAL", "facility_manager")
                st.rerun()
            if c2.button("Keep server value", key=f"srv_{o['id']}"):
                resolve_conflict(o["id"], "KEEP_SERVER", "facility_manager")
                st.rerun()
