package trust40.k4case

import java.time.LocalDateTime

import akka.actor.{Actor, ActorRef, Props}
import akka.event.Logging
import trust40.k4case.Resolver.{Resolve, ResolverResult}

import scala.collection.mutable
import Simulation.{Notifications, Permissions, Reset, Step}

object Enforcer {
  def props(resolver: ActorRef) = Props(new Enforcer(resolver))

  final case class Events(events: List[ScenarioEvent])
}


class Enforcer(val resolver: ActorRef) extends Actor {
  import Enforcer._

  private val log = Logging(context.system, this)

  private var eventsSinceLastResolve = mutable.ListBuffer.empty[ScenarioEvent]
  private var resolving: Boolean = _
  private var resolvingCurrentTime: LocalDateTime = _

  private var nextStepNo: Int = _
  private var resolveNeeded: Boolean = _

  private var currentEpoch: Int = _
  private var currentPermissions: List[Permission] = _
  private var currentRejectedPermissions: Map[AllowPermission, List[DenyPermission]] = _
  private val currentNotifications: mutable.Set[ComponentNotification] = mutable.Set.empty[ComponentNotification]

  private def processReset(epoch: Int): Unit = {
    currentEpoch = epoch

    eventsSinceLastResolve.clear()
    resolving = false
    resolvingCurrentTime = null
    resolveNeeded = false
    nextStepNo = 0

    currentPermissions = List()
    currentRejectedPermissions = Map()
    currentNotifications.clear()
  }

  private def processEvents(events: List[ScenarioEvent]): Unit = {
    eventsSinceLastResolve ++= events
  }

  private def processResolverResult(permissions: List[Permission], notifications: List[ComponentNotification], rejectedPermissions: Map[AllowPermission, List[DenyPermission]]): Unit = {
    resolving = false
    currentPermissions = permissions
    currentRejectedPermissions = rejectedPermissions

    for (notif <- notifications) {
      currentNotifications += notif
    }

    context.parent ! Permissions(currentEpoch, currentPermissions, currentRejectedPermissions)
    context.parent ! Notifications(currentEpoch, currentNotifications.toList)
  }

  private def processStep(currentTime: LocalDateTime): Unit = {
    val stepNo = nextStepNo
    nextStepNo += 1

    if (stepNo % 6 == 0) {
      resolveNeeded = true
    }

    if (resolveNeeded && !resolving) {
      resolving = true
      resolvingCurrentTime = currentTime
      resolveNeeded = false

      val events = eventsSinceLastResolve.sortWith((ev1, ev2) => ev1.timestamp.isBefore(ev2.timestamp)).toList
      eventsSinceLastResolve.clear()
      resolver ! Resolve(currentTime, events)
    }
  }

  def receive = {
    case Reset(epoch) => processReset(epoch)
    case Events(events) => processEvents(events)
    case Step(currentTime) => processStep(currentTime)

    case ResolverResult(epoch, permissions, notifications, rejectedPermissions) =>
      if (currentEpoch == epoch) processResolverResult(permissions, notifications, rejectedPermissions)
  }

}

