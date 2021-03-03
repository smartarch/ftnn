package trust40.k4case

import java.time.format.DateTimeFormatter
import java.time.{Duration, LocalDateTime, ZoneOffset}

import akka.actor.{Actor, ActorRef, Props, Timers}
import akka.event.Logging
import akka.pattern.{ask, pipe}
import akka.util.Timeout

import scala.collection.mutable
import scala.concurrent.duration._

case class WorkerState(position: Position, hasHeadGear: Boolean, standbyFor: Option[String])
case class SimulationState(time: String, playState: Simulation.State.State, workers: Map[String, WorkerState], permissions: List[(String, String, String)], rejectedPermissions: List[(String, String, String, List[(String, String, String, String)])])
case class AccessResult(result: String)
case class ValidateResult(allowed: Boolean)

object Simulation {
  def props() = Props(new Simulation())

  case object Play
  case object Pause
  case object Reset
  case object Status
  final case class Access(workerId: String)
  final case class Validate(subjectId: String, verb: String, objectId: String)

  final case class Reset(epoch: Int)
  final case class Events(epoch: Int, events: List[ScenarioEvent])
  final case class Permissions(epoch: Int, permissions: List[Permission], rejectedPermissions: Map[AllowPermission, List[DenyPermission]])
  final case class Notifications(epoch: Int, notifications: List[ComponentNotification])
  final case class Step(currentTime: LocalDateTime)
  final case class WorkerStep(currentTime: LocalDateTime, notifications: List[(String, List[String])])
  final case class WorkerAccess(workerPermissions: List[Permission])

  private case object TickTimer
  private case object Tick

  object State extends Enumeration {
    type State = Value

    // This has to be aligned with visualizer/client/src/FactoryMap.js
    val START = Value(0)
    val PLAYING = Value(1)
    val PAUSED = Value(2)
    val END = Value(3)
  }
}

class Simulation() extends Actor with Timers {
  import Simulation.State._
  import Simulation._

  private val log = Logging(context.system, this)

  private var state = START
  private var currentTime: LocalDateTime = _

  private val startTime = LocalDateTime.parse("2018-12-03T08:00:00")
  private val endTime = startTime plus Duration.ofHours(10)

  private val scenarioSpec = TestScenario.createScenarioSpec(workersPerWorkplaceCount=3, workersOnStandbyCount=1, workersLateCount=0, startTime)

  private val resolver = context.actorOf(Resolver.props(scenarioSpec), name = "resolver")
  private val enforcer: ActorRef = context.actorOf(Enforcer.props(resolver), name = "enforcer")

  private var workers = mutable.ListBuffer.empty[ActorRef]
  // foremen
  workers += context.actorOf(SimulatedWorkerInShift.props(s"A-foreman", "A", "F", startTime), name = s"A-foreman")
  workers += context.actorOf(SimulatedWorkerInShift.props(s"B-foreman", "B", "F", startTime), name = s"B-foreman")
  workers += context.actorOf(SimulatedWorkerInShift.props(s"C-foreman", "C", "F", startTime), name = s"C-foreman")

  {
    var remainingLateWorkers = scenarioSpec.workersLateCount

    def addWorker(wpId: String, idx: Int): Unit = {
      if (wpId == "A" && idx == 1) {
        workers += context.actorOf(UserControlledWorkerInShift.props(f"$wpId%s-worker-$idx%03d", wpId, idx.toString, startTime), name = f"$wpId%s-worker-$idx%03d")
      } else if (remainingLateWorkers > 0) {
        workers += context.actorOf(SimulatedLateWorkerInShift.props(f"$wpId%s-worker-$idx%03d", wpId, idx.toString, startTime), name = f"$wpId%s-worker-$idx%03d")
        remainingLateWorkers -= 1
      } else {
        workers += context.actorOf(SimulatedWorkerInShift.props(f"$wpId%s-worker-$idx%03d", wpId, idx.toString, startTime), name = f"$wpId%s-worker-$idx%03d")
      }
    }

    // workers
    for (idx <- 1 to scenarioSpec.workersPerWorkplaceCount) {
      addWorker("A", idx)
      addWorker("B", idx)
      addWorker("C", idx)
    }
  }

  for (idx <- 1 to scenarioSpec.workersOnStandbyCount) {
    workers += context.actorOf(SimulatedStandbyInShift.props(f"standby-$idx%03d", idx.toString, startTime), name = f"standby-$idx%03d")
  }

  private val workerStates = mutable.HashMap.empty[String, WorkerState]
  private var currentPermissions: List[Permission] = _
  private var currentRejectedPermissions: Map[AllowPermission, List[DenyPermission]] = _
  private var currentNotifications: List[ComponentNotification] = _

  private var currentEpoch = 0

  processReset()


  private def processReset(): Unit = {
    currentEpoch += 1

    state = START
    timers.cancel(TickTimer)

    currentTime = null
    workerStates.clear()
    currentPermissions = List()
    currentRejectedPermissions = Map()
    currentNotifications = List()

    resolver ! Reset(currentEpoch)
    enforcer ! Reset(currentEpoch)

    for (worker <- workers) {
      worker ! Reset(currentEpoch)
    }

    processTick()
  }

  private def processPlay(): Unit = {
    if (state == START || state == PAUSED) {
      state = PLAYING
      timers.startPeriodicTimer(TickTimer, Tick, 100 millis)
    }
  }


  private def processPause(): Unit = {
    if (state == PLAYING) {
      state = PAUSED
      timers.cancel(TickTimer)
    }
  }

  private def processTick(): Unit = {
    if (currentTime == null) {
      currentTime = startTime
    } else {
      currentTime = currentTime plusSeconds(10)
    }

    for (worker <- workers) {
      val workerName = worker.path.name
      val notifs = currentNotifications.collect {
        case ComponentNotification(`workerName`, action, params) => (action, params)
      }
      worker ! WorkerStep(currentTime, notifs)
    }

    enforcer ! Step(currentTime)

    if (!currentTime.isBefore(endTime)) {
      state = END
      timers.cancel(TickTimer)
    }
  }

  private def processStatus(): Unit = {
    sender() ! SimulationState(
      currentTime.atZone(ZoneOffset.UTC).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME),
      state,
      workerStates.toMap,
      currentPermissions.collect({
        case AllowPermission(subj, verb, obj) => (subj, verb, obj)
      }),
      currentRejectedPermissions.collect({
        case (AllowPermission(allowSubj, allowVerb, allowObj), (denyRules: List[DenyPermission])) => (allowSubj, allowVerb, allowObj, denyRules.collect({case DenyPermission(denySubj, denyVerb, denyObj, denyLevel) => (denySubj, denyVerb, denyObj, denyLevel.toString)}))
      }).toList
    )
  }

  private def processAccess(workerId: String): Unit = {
    import context.dispatcher
    implicit val timeout = Timeout(1 second)

    workers.find(worker => worker.path.name == workerId) match {
      case Some(worker) =>
        val workerPermissions = currentPermissions.filter({
          case AllowPermission(`workerId`, verb, obj) => true
          case _ => false
        })

        (worker ? WorkerAccess(workerPermissions)).pipeTo(sender())

      case None =>
        sender() ! AccessResult("none")
    }
  }

  private def processValidate(subjectId: String, verb: String, objectId: String): Unit = {
    sender() ! ValidateResult(currentPermissions.contains(AllowPermission(subjectId, verb, objectId)))
  }


  private def processEvents(events: List[ScenarioEvent]): Unit = {
    for (event <- events) {
      val oldState = workerStates.getOrElse(event.person, WorkerState(null, false, None))

      var hasHeadGear = oldState.hasHeadGear
      var standbyFor = oldState.standbyFor

      if (event.eventType == "retrieve-head-gear") {
        hasHeadGear = true
      }

      if (event.eventType == "return-head-gear") {
        hasHeadGear = false
      }

      if (event.eventType == "take-over") {
        standbyFor = Some(event.params(0))
      }

      workerStates(event.person) = WorkerState(event.position, hasHeadGear, standbyFor)
    }

    enforcer ! Enforcer.Events(events)
  }

  private def processPermissions(permissions: List[Permission], rejectedPermissions: Map[AllowPermission, List[DenyPermission]]): Unit = {
    currentPermissions = permissions
    currentRejectedPermissions = rejectedPermissions
  }

  private def processNotifications(notifications: List[ComponentNotification]): Unit = {
    currentNotifications = notifications
  }


  def receive = {
    case Play => processPlay()
    case Pause => processPause()
    case Reset => processReset()
    case Status => processStatus()
    case Access(workerId) => processAccess(workerId)
    case Validate(subjectId, verb, objectId) => processValidate(subjectId, verb, objectId)

    case Tick => processTick()

    case Events(epoch, events) =>
      if (epoch == currentEpoch) {
        processEvents(events)
      }

    case Notifications(epoch, notifications) =>
      if (epoch == currentEpoch) {
        processNotifications(notifications)
      }

    case Permissions(epoch, permissions, rejectedPermissions) =>
      if (epoch == currentEpoch) {
        processPermissions(permissions, rejectedPermissions)
      }
  }
}

