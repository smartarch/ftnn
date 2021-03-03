package trust40.enforcer.tcof
import collection.mutable

abstract class Notification(private val notifType: String, private val notifParams: List[String]) {
  def getType: String = notifType
  def getParams: List[String] = notifParams
}

trait Notifiable {
  private[tcof] val _notificationsReceived = mutable.Set.empty[Notification]

  def notified(notification: Notification): Boolean = _notificationsReceived.contains(notification)
  def notifiedExt(notificationFilter: PartialFunction[Notification, Boolean]): Boolean = _notificationsReceived.exists(notif => if (notificationFilter.isDefinedAt(notif)) notificationFilter(notif) else false)
  def notify(notification: Notification): Unit = _notificationsReceived += notification
}
