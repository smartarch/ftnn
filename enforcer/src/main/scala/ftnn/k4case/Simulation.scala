package ftnn.k4case

import akka.actor.{Actor, Props, Stash, Timers}
import akka.event.Logging
import ftnn.k4case.Resolver.{ResolverReset, ResolverResetDone, ResolverResult, ResolverStep}

import java.time.format.DateTimeFormatter
import java.time.{Duration, LocalDateTime, ZoneOffset}
import scala.collection.mutable
import scala.concurrent.duration._
import scala.language.postfixOps

case class WorkerState(position: Position, hasHeadGear: Boolean, standbyFor: Option[String])
case class AccessResult(result: String)
case class SimulationState(time: String, playState: Simulation.State.State, workers: Map[String, WorkerState], permissions: List[(String, String, String)])


object Simulation {
  def props(randSeed: Int, traceFileBase: String) = Props(new Simulation(randSeed, traceFileBase))

  final case class Play(tickIntervalMs: Int)
  case object Pause
  case object Reset
  case object Status

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

class Simulation(val randSeed: Int, val traceFileBase: String) extends Actor with Timers with Stash {
  import Simulation.State._
  import Simulation._

  private val log = Logging(context.system, this)

  private var state = START
  private var currentTime: LocalDateTime = _

  private var ticksToResolution = 0
  private var eventsSinceLastResolve = mutable.ListBuffer.empty[ScenarioEvent]

  private val startTime = LocalDateTime.parse("2018-12-03T08:00:00")
  private val endTime = startTime plus Duration.ofHours(10)

  private val scenarioSpec = Scenario.createScenarioSpec(workersPerWorkplaceCount=3, workersOnStandbyCount=2, workersLateCount=3, startTime)

  private val resolver = context.actorOf(Resolver.props(scenarioSpec, traceFileBase), name = "resolver")

  private var workers = mutable.ListBuffer.empty[AbstractSimulatedWorker]

  private var tickIntervalMs = 0

  // foremen
  workers += new SimulatedWorkerInShift(randSeed, s"A-foreman", "A", "F", startTime)
  workers += new SimulatedWorkerInShift(randSeed, s"B-foreman", "B", "F", startTime)
  workers += new SimulatedWorkerInShift(randSeed, s"C-foreman", "C", "F", startTime)

  {
    var remainingLateWorkers = scenarioSpec.workersLateCount

    def addWorker(wpId: String, idx: Int): Unit = {
      if (remainingLateWorkers > 0) {
        workers += new SimulatedLateWorkerInShift(randSeed, f"$wpId%s-worker-$idx%03d", wpId, idx.toString, startTime)
        remainingLateWorkers -= 1
      } else {
        workers += new SimulatedWorkerInShift(randSeed, f"$wpId%s-worker-$idx%03d", wpId, idx.toString, startTime)
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
    workers += new SimulatedStandbyInShift(randSeed, f"A-standby-$idx%03d", idx.toString, startTime)
    workers += new SimulatedStandbyInShift(randSeed, f"B-standby-$idx%03d", idx.toString, startTime)
    workers += new SimulatedStandbyInShift(randSeed, f"C-standby-$idx%03d", idx.toString, startTime)
  }

  private val workerStates = mutable.HashMap.empty[String, WorkerState]
  private var currentPermissions: List[Permission] = _
  private val currentNotifications: mutable.Set[ComponentNotification] = mutable.Set.empty[ComponentNotification]

  performReset()
  performStep()

  private def performReset(): Unit = {
    state = START
    timers.cancel(TickTimer)

    currentTime = startTime
    workerStates.clear()
    currentPermissions = List()
    currentNotifications.clear()

    eventsSinceLastResolve.clear()
    ticksToResolution = 0

    for (worker <- workers) {
      worker.reset()
    }
  }

  private def processReset(): Unit = {
    performReset()

    resolver ! ResolverReset
    context.become(awaitResolverResetDone)
    unstashAll()
  }

  private def processPlay(tickIntervalMs: Int): Unit = {
    log.debug(s"processPlay: state=${state}")
    if (state == START || state == PAUSED) {
      state = PLAYING

      this.tickIntervalMs = tickIntervalMs
      if (tickIntervalMs > 0) {
        timers.startTimerAtFixedRate(TickTimer, Tick, tickIntervalMs millis)
      } else {
        self ! Tick
      }
    }
  }


  private def processPause(): Unit = {
    if (state == PLAYING) {
      state = PAUSED

      if (tickIntervalMs > 0) {
        timers.cancel(TickTimer)
      }
    }
  }

  private def processTick(): Unit = {
    log.info(s"processTick: time=${currentTime} state=${state} ticksToResolution=${ticksToResolution}")
    assert(state == PLAYING)

    currentTime = currentTime plusSeconds (10)

    if (ticksToResolution == 0) {
      context.become(awaitResolverResult)
      unstashAll()
    } else {
      performStep()
    }
  }

  private def processResolverResult(permissions: List[Permission], notifications: List[ComponentNotification]): Unit = {
    log.info(s"processResolverResult: time=${currentTime} state=${state} ticksToResolution=${ticksToResolution}")
    context.become(receive)
    unstashAll()

    currentPermissions = permissions
    currentNotifications ++= notifications
    performStep()
  }

  private def processResolverResetDone(): Unit = {
    context.become(receive)
    unstashAll()

    performStep()
  }

  private def performStep(): Unit = {
    log.info(s"performStep: time=${currentTime} state=${state} ticksToResolution=${ticksToResolution}")
    if (!currentTime.isBefore(endTime)) {
      state = END
      if (tickIntervalMs > 0) {
        timers.cancel(TickTimer)
      }

    } else {
      for (worker <- workers) {
        val workerName = worker.name
        val notifs = currentNotifications.collect {
          case ComponentNotification(`workerName`, action, params) => (action, params)
        }

        val events = worker.step(currentTime, notifs.toList)

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

        eventsSinceLastResolve ++= events
      }

      if (ticksToResolution == 0) {
        val events = eventsSinceLastResolve.sortWith((ev1, ev2) => ev1.timestamp.isBefore(ev2.timestamp)).toList
        eventsSinceLastResolve.clear()

        resolver ! ResolverStep(currentTime, events)
        ticksToResolution = 5
      } else {
        ticksToResolution -= 1
      }

      if (tickIntervalMs == 0 && state == PLAYING) {
        self ! Tick
      }
    }
  }

  private def processStatus(): Unit = {
    sender() ! SimulationState(
      currentTime.atZone(ZoneOffset.UTC).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME),
      state,
      workerStates.toMap,
      currentPermissions.collect({
        case AllowPermission(subj, verb, obj) => (subj, verb, obj)
      })
    )
  }

  def receive = {
    case Play(tickIntervalMs) => processPlay(tickIntervalMs)
    case Pause => processPause()
    case Reset => processReset()
    case Status => processStatus()

    case Tick => processTick()

    case msg =>
      log.debug(s"receive: stashing msg ${msg}")
      stash()
  }

  def awaitResolverResult: Receive = {
    case ResolverResult(permissions, notifications) => processResolverResult(permissions, notifications)

    case msg =>
      log.debug(s"awaitResolverResult: stashing msg ${msg}")
      stash()
  }

  def awaitResolverResetDone: Receive = {
    case ResolverResetDone => processResolverResetDone()

    case msg =>
      log.debug(s"awaitResolverResetDone: discarding msg ${msg}")
  }
}

