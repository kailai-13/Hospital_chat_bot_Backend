from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import re

# In-memory fallback storage
in_memory_storage = {
    'chat_history': [],
    'appointment_requests': [],
    'admin_notifications': []
}

# Firestore optional (not required). Keeping None to avoid breaking existing setup.
db = None
FIREBASE_INITIALIZED = False

# Pydantic models (scoped to admin API)
class AdminChatMessage(BaseModel):
    message: str
    user_role: str = "patient"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    phone_number: Optional[str] = None

class AdminChatResponse(BaseModel):
    response: str
    timestamp: str
    is_appointment_request: bool = False
    appointment_id: Optional[str] = None

class AppointmentRequest(BaseModel):
    user_name: str
    phone_number: str
    preferred_date: str
    preferred_time: str
    reason: str
    user_role: str = "patient"

class AppointmentAction(BaseModel):
    appointment_id: str
    action: str  # "accept" or "reject"
    admin_notes: Optional[str] = None

router = APIRouter()

# Utility functions (available to import in main.py)

def save_chat_history(user_id: str, user_role: str, user_name: str,
                      message: str, response: str, is_appointment: bool = False):
    chat_data = {
        'id': str(uuid.uuid4()),
        'user_id': user_id or 'anonymous',
        'user_role': user_role,
        'user_name': user_name or 'Anonymous User',
        'message': message,
        'response': response,
        'is_appointment_request': is_appointment,
        'created_at': datetime.now().isoformat()
    }
    if db is not None and FIREBASE_INITIALIZED:
        try:
            doc_ref = db.collection('chat_history').document(chat_data['id'])
            doc_ref.set(chat_data)
            return True
        except Exception:
            pass
    in_memory_storage['chat_history'].append(chat_data)
    return True


def save_admin_notification(title: str, message: str, notification_type: str = "info"):
    notification_data = {
        'id': str(uuid.uuid4()),
        'title': title,
        'message': message,
        'type': notification_type,
        'read': False,
        'created_at': datetime.now().isoformat()
    }
    if db is not None and FIREBASE_INITIALIZED:
        try:
            db.collection('admin_notifications').add(notification_data)
            return True
        except Exception:
            pass
    in_memory_storage['admin_notifications'].append(notification_data)
    return True


def save_appointment_request(user_name: str, phone_number: str,
                             preferred_date: str, preferred_time: str,
                             reason: str, user_role: str, original_message: str):
    appointment_id = str(uuid.uuid4())
    appointment_data = {
        'appointment_id': appointment_id,
        'user_name': user_name,
        'phone_number': phone_number,
        'preferred_date': preferred_date,
        'preferred_time': preferred_time,
        'reason': reason,
        'user_role': user_role,
        'original_message': original_message,
        'status': 'pending',
        'admin_notes': '',
        'created_at': datetime.now().isoformat()
    }
    if db is not None and FIREBASE_INITIALIZED:
        try:
            db.collection('appointment_requests').document(appointment_id).set(appointment_data)
            save_admin_notification(
                "📅 New Appointment Request",
                f"Patient: {user_name}\nPhone: {phone_number}\nDate: {preferred_date}\nTime: {preferred_time}",
                "appointment_request"
            )
            return appointment_id
        except Exception:
            pass
    in_memory_storage['appointment_requests'].append(appointment_data)
    save_admin_notification(
        "📅 New Appointment Request",
        f"Patient: {user_name}\nPhone: {phone_number}\nDate: {preferred_date}\nTime: {preferred_time}",
        "appointment_request"
    )
    return appointment_id


def get_all_chat_history(user_role: Optional[str] = None, limit: int = 100):
    if db is not None and FIREBASE_INITIALIZED:
        try:
            query = db.collection('chat_history')
            if user_role and user_role != 'all':
                query = query.where('user_role', '==', user_role)
            docs = query.limit(limit).stream()
            history = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                history.append(data)
            history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return history
        except Exception:
            pass
    history = in_memory_storage['chat_history'].copy()
    if user_role and user_role != 'all':
        history = [chat for chat in history if chat.get('user_role') == user_role]
    history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return history[:limit]


def get_appointment_requests(status: Optional[str] = None):
    if db is not None and FIREBASE_INITIALIZED:
        try:
            query = db.collection('appointment_requests')
            if status:
                query = query.where('status', '==', status)
            docs = query.stream()
            appointments = [doc.to_dict() for doc in docs]
            appointments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return appointments
        except Exception:
            pass
    appointments = in_memory_storage['appointment_requests'].copy()
    if status:
        appointments = [apt for apt in appointments if apt.get('status') == status]
    appointments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return appointments


def update_appointment_status(appointment_id: str, action: str, admin_notes: str = ""):
    new_status = 'accepted' if action == 'accept' else 'rejected'
    if db is not None and FIREBASE_INITIALIZED:
        try:
            doc_ref = db.collection('appointment_requests').document(appointment_id)
            doc = doc_ref.get()
            if doc.exists:
                appointment_data = doc.to_dict()
                doc_ref.update({
                    'status': new_status,
                    'admin_notes': admin_notes,
                    'updated_at': datetime.now().isoformat()
                })
                save_admin_notification(
                    f"📋 Appointment {new_status.capitalize()}",
                    f"Patient: {appointment_data.get('user_name', 'Unknown')}\n"
                    f"Phone: {appointment_data.get('phone_number', 'Not provided')}\n"
                    f"Status: {new_status}\n"
                    f"Notes: {admin_notes}",
                    "appointment_action"
                )
                return True
        except Exception:
            pass
    for appointment in in_memory_storage['appointment_requests']:
        if appointment.get('appointment_id') == appointment_id:
            appointment['status'] = new_status
            appointment['admin_notes'] = admin_notes
            appointment['updated_at'] = datetime.now().isoformat()
            save_admin_notification(
                f"📋 Appointment {new_status.capitalize()}",
                f"Patient: {appointment.get('user_name', 'Unknown')}\n"
                f"Phone: {appointment.get('phone_number', 'Not provided')}\n"
                f"Status: {new_status}\n"
                f"Notes: {admin_notes}",
                "appointment_action"
            )
            return True
    return False


def get_admin_notifications(limit: int = 20):
    if db is not None and FIREBASE_INITIALIZED:
        try:
            query = db.collection('admin_notifications')
            docs = query.limit(limit).stream()
            notifications = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                notifications.append(data)
            notifications.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return notifications
        except Exception:
            pass
    notifications = in_memory_storage['admin_notifications'].copy()
    notifications.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return notifications[:limit]


def mark_notification_read(notification_id: str):
    if db is not None and FIREBASE_INITIALIZED:
        try:
            doc_ref = db.collection('admin_notifications').document(notification_id)
            doc_ref.update({
                'read': True,
                'read_at': datetime.now().isoformat()
            })
            return True
        except Exception:
            pass
    for notification in in_memory_storage['admin_notifications']:
        if notification.get('id') == notification_id:
            notification['read'] = True
            notification['read_at'] = datetime.now().isoformat()
            return True
    return False

# Intent and detail extraction

def detect_appointment_intent(message: str) -> bool:
    appointment_keywords = [
        'appointment', 'book', 'schedule', 'meet', 'doctor visit',
        'consultation', 'checkup', 'visit doctor', 'see doctor',
        'reserve', 'slot', 'available time'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in appointment_keywords)


def extract_appointment_details(message: str) -> dict:
    details = {'date': None, 'time': None, 'reason': None}
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b(today|tomorrow|next week|next month)\b'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            details['date'] = match.group(0)
            break
    time_patterns = [
        r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\b',
        r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',
        r'\b(morning|afternoon|evening)\b'
    ]
    for pattern in time_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            details['time'] = match.group(0)
            break
    reason_indicators = ['for', 'regarding', 'about', 'because']
    for indicator in reason_indicators:
        if indicator in message.lower():
            parts = message.lower().split(indicator, 1)
            if len(parts) > 1:
                details['reason'] = parts[1].strip()[:200]
                break
    return details

# Admin endpoints

@router.get("/admin/chat-history")
async def admin_get_chat_history(user_role: Optional[str] = None, limit: int = 100):
    try:
        history = get_all_chat_history(user_role=user_role, limit=limit)
        return {
            "history": history,
            "count": len(history),
            "filtered_by": user_role or "all",
            "storage_mode": "memory" if db is None else "firestore"
        }
    except Exception as e:
        return {"history": [], "count": 0, "filtered_by": user_role or "all", "storage_mode": "error", "error": str(e)}


@router.get("/admin/appointments")
async def admin_get_appointments(status: Optional[str] = None):
    try:
        appointments = get_appointment_requests(status=status)
        return {
            "appointments": appointments,
            "count": len(appointments),
            "filtered_by": status or "all",
            "storage_mode": "memory" if db is None else "firestore"
        }
    except Exception as e:
        return {"appointments": [], "count": 0, "filtered_by": status or "all", "storage_mode": "error", "error": str(e)}


@router.post("/admin/appointments/action")
async def admin_handle_appointment_action(action: AppointmentAction):
    try:
        if action.action not in ['accept', 'reject']:
            raise HTTPException(status_code=400, detail="Action must be 'accept' or 'reject'")
        success = update_appointment_status(
            appointment_id=action.appointment_id,
            action=action.action,
            admin_notes=action.admin_notes or ""
        )
        if success:
            return {"message": f"Appointment {action.action}ed successfully", "appointment_id": action.appointment_id, "status": action.action + "ed"}
        else:
            raise HTTPException(status_code=404, detail="Appointment not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating appointment: {str(e)}")


@router.get("/admin/notifications")
async def admin_get_admin_notifications(limit: int = 20):
    try:
        notifications = get_admin_notifications(limit=limit)
        return {
            "notifications": notifications,
            "count": len(notifications),
            "storage_mode": "memory" if db is None else "firestore"
        }
    except Exception:
        return {"notifications": [], "count": 0, "storage_mode": "error"}


@router.post("/admin/notifications/mark-read")
async def admin_mark_notification_read(notification_id: str):
    try:
        success = mark_notification_read(notification_id)
        if success:
            return {"success": True, "message": "Notification marked as read"}
        else:
            raise HTTPException(status_code=500, detail="Failed to mark notification as read")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking notification: {str(e)}")


@router.get("/admin/statistics")
async def admin_get_admin_statistics():
    try:
        all_chats = get_all_chat_history(limit=1000)
        pending_appointments = get_appointment_requests(status='pending')
        accepted_appointments = get_appointment_requests(status='accepted')
        rejected_appointments = get_appointment_requests(status='rejected')
        role_counts: Dict[str, int] = {}
        for chat in all_chats:
            role = chat.get('user_role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        return {
            "total_conversations": len(all_chats),
            "pending_appointments": len(pending_appointments),
            "accepted_appointments": len(accepted_appointments),
            "rejected_appointments": len(rejected_appointments),
            "conversations_by_role": role_counts,
            "timestamp": datetime.now().isoformat(),
            "storage_mode": "memory" if db is None else "firestore"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")