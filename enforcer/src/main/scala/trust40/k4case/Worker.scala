package trust40.k4case

import java.time.temporal.ChronoUnit
import java.time.{Duration, LocalDateTime}

import akka.actor.{Actor, Props}
import akka.event.Logging
import trust40.k4case.Simulation.{Events, Reset, WorkerAccess, WorkerStep}

import scala.collection.mutable
import scala.util.Random

object AbstractSimulatedWorker {
  private val random = new Random(42)
}

abstract class AbstractSimulatedWorker(val person: String, val startPosition: Position, val startTime: LocalDateTime) extends Actor {
  import AbstractSimulatedWorker.random
  protected val log = Logging(context.system, this)

  private abstract class Action(val startTime: LocalDateTime, val duration: Duration, val startPosition: Position, val targetPosition: Position) {
    def getEvents(currentTime: LocalDateTime): List[ScenarioEvent] = List()
    def isDue(currentTime: LocalDateTime) = !currentTime.isBefore(startTime)
    def isOver(currentTime: LocalDateTime) = !currentTime.isBefore(startTime.plus(duration))
  }

  private class InitAction() extends Action(startTime, Duration.ZERO, startPosition, startPosition) {
    override def toString: String = s"InitAction($startTime, $startPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(startTime, "init", person, startPosition, List()))
  }

  private class MoveAction(startTime: LocalDateTime, duration: Duration, startPosition: Position, targetPosition: Position) extends Action(startTime, duration, startPosition, targetPosition) {
    override def toString: String = s"MoveAction($startTime, $duration, $startPosition, $targetPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = {
      var nsSinceStart = ChronoUnit.NANOS.between(startTime, currentTimestamp)
      val nsTotal = duration.toNanos

      if (nsSinceStart > nsTotal) {
        nsSinceStart = nsTotal
      }

      val alpha = nsSinceStart.toDouble / nsTotal

      val x = startPosition.x * (1 - alpha) + targetPosition.x * alpha
      val y = startPosition.y * (1 - alpha) + targetPosition.y * alpha

      List(ScenarioEvent(startTime.plusNanos(nsSinceStart), "move", person, Position(x, y), List()))
    }
  }

  private class AccessDoorAction(timestamp: LocalDateTime, doorPosition: Position) extends Action(timestamp, Duration.ZERO, doorPosition, doorPosition) {
    override def toString: String = s"AccessDoorAction($timestamp, $doorPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "access-door", person, doorPosition, List()))
  }

  private class TakeOverAction(timestamp: LocalDateTime, position: Position, replacedWorkerId: String) extends Action(timestamp, Duration.ZERO, position, position) {
    override def toString: String = s"TakeOverAction($timestamp, $replacedWorkerId)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "take-over", person, position, List(replacedWorkerId)))
  }

  private class RetrieveHeadGearAction(timestamp: LocalDateTime, dispenserPosition: Position) extends Action(timestamp, Duration.ZERO, dispenserPosition, dispenserPosition) {
    override def toString: String = s"RetrieveHeadGearAction($timestamp, $dispenserPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "retrieve-head-gear", person, dispenserPosition, List()))
  }

  private class ReturnHeadGearAction(timestamp: LocalDateTime, dispenserPosition: Position) extends Action(timestamp, Duration.ZERO, dispenserPosition, dispenserPosition) {
    override def toString: String = s"ReturnHeadGearAction($timestamp, $dispenserPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "return-head-gear", person, dispenserPosition, List()))
  }

  private class WaitAction(timestamp: LocalDateTime, duration: Duration, position: Position) extends Action(timestamp, duration, position, position) {
    override def toString: String = s"WaitAction($timestamp, $duration, $position)"
  }

  private val futureActions = mutable.ListBuffer.empty[Action]
  private var currentAction: Action = _

  protected var currentTime: LocalDateTime = _
  protected var currentPosition: Position = _

  protected var currentNotifications: List[(String, List[String])] = List()

  protected var currentEpoch: Int = _

  private def generateEvents(time: LocalDateTime): List[ScenarioEvent] = {
    val events = mutable.ListBuffer.empty[ScenarioEvent]

    def addEventsAndDropAction(): Unit = {
      events ++= currentAction.getEvents(time)

      if (futureActions.isEmpty) {
        generateActions()
      }

      currentAction = null
    }


    if (currentAction == null && futureActions.isEmpty) {
      currentTime = time
      generateActions()
    }

    if (currentAction != null && currentAction.isOver(time)) {
      currentTime = time
      addEventsAndDropAction()
    }

    while (futureActions.nonEmpty && currentAction == null) {
      currentAction = futureActions.head
      currentTime = currentAction.startTime.plus(currentAction.duration)
      currentPosition = currentAction.targetPosition

      futureActions.remove(0)

      if (currentAction.isDue(time) && currentAction.isOver(time)) {
        addEventsAndDropAction()
      }
    }

    if (currentAction != null && currentAction.isDue(time)) {
      events ++= currentAction.getEvents(time)
    }

    events.toList
  }

  private def lastActionTime = if (futureActions.nonEmpty) futureActions.last.startTime.plus(futureActions.last.duration) else currentTime
  private def lastActionPosition = if (futureActions.nonEmpty) futureActions.last.targetPosition else currentPosition

  protected def move(id: String, maxPace: Double = 9 /* seconds / position unit */, minPace: Double = 11): Unit = {
    val targetPosition = FactoryMap(id)
    val startPosition = lastActionPosition
    val xDistance = startPosition.x - targetPosition.x
    val yDistance = startPosition.y - targetPosition.y
    val distance = Math.sqrt(xDistance * xDistance + yDistance * yDistance)
    val duration = Duration.ofMillis((distance * (maxPace + random.nextDouble() * (minPace - maxPace)) * 1000).toInt)

    futureActions += new MoveAction(lastActionTime, duration, startPosition, targetPosition)
  }

  protected def accessDoor(): Unit = futureActions += new AccessDoorAction(lastActionTime, lastActionPosition)

  protected def takeOver(replacedWorkerId: String): Unit = futureActions += new TakeOverAction(lastActionTime, lastActionPosition, replacedWorkerId)

  protected def retrieveHeadGear(): Unit = futureActions += new RetrieveHeadGearAction(lastActionTime, lastActionPosition)
  protected def returnHeadGear(): Unit = futureActions += new ReturnHeadGearAction(lastActionTime, lastActionPosition)

  protected def waitRandom(minDuration: Duration, maxDuration: Duration): Unit = {
    val duration = minDuration plusSeconds random.nextInt((maxDuration minus minDuration).getSeconds().asInstanceOf[Int])
    futureActions += new WaitAction(lastActionTime, duration, lastActionPosition)
  }

  protected def waitTillAfterStart(durationFromStart: Duration): Unit = {
    val startTime = lastActionTime
    val endTime = this.startTime plus durationFromStart
    if (startTime.isBefore(endTime)) {
      val duration = Duration.between(startTime, endTime)
      futureActions += new WaitAction(startTime, duration, lastActionPosition)
    }
  }

  protected implicit class EventDuration(value: Int) {
    def seconds = Duration.ofSeconds(value)
    def minutes = Duration.ofMinutes(value)
    def hours = Duration.ofHours(value)
  }

  protected def generateActions() {}
  protected def generateInitialActions() {}

  protected def processAccess(workerPermissions: List[Permission]): Unit = {
    sender() ! AccessResult("none")
  }

  private def processStep(currentTime: LocalDateTime, notifications: List[(String, List[String])]): Unit = {
    currentNotifications = notifications
    val events = generateEvents(currentTime)
    context.parent ! Events(currentEpoch, events)
  }

  private def processReset(epoch: Int): Unit = {
    currentEpoch = epoch

    futureActions.clear()
    currentAction = null

    currentTime = startTime
    currentPosition = startPosition

    futureActions += new InitAction()
    generateInitialActions()
  }

  def receive = {
    case WorkerStep(currentTime, notifications) => processStep(currentTime, notifications)
    case Reset(epoch) => processReset(epoch)
    case WorkerAccess(workerPermissions) => processAccess(workerPermissions)
  }
}


object SimulatedWorkerInShift {
  def props(person: String, wpId: String, inShiftId: String, startTime: LocalDateTime) = Props(new SimulatedWorkerInShift(person, wpId, inShiftId, startTime))
}

// Pos-InWorkPlace1-1, Pos-JunctionToWorkPlace1and2, Pos-InFrontOfMainGate, Pos-Init3-2, Pos-Init2-1, Pos-InWorkPlace2-3, Pos-Gate1,
// Pos-InFrontOfGate2, Pos-InWorkPlace3-3, Pos-BottomRight, Pos-Init1-3, Pos-InWorkPlace3-1, Pos-InFrontOfGate3, Pos-Init3-1, Pos-InWorkPlace2-1, Pos-Gate3,
// Pos-Dispenser, Pos-MainGate, Pos-InWorkPlace1-3, Pos-Init1-2, Pos-Init2-3, Pos-InWorkPlace3-2, Pos-InitStandby-1, Pos-TopLeft, Pos-InWorkPlace1-2, Pos-Init1-1,
// Pos-InWorkPlace2-2, Pos-Init2-2, Pos-Gate2, Pos-InFrontOfGate1, Pos-Init3-3

class SimulatedWorkerInShift(person: String, val wpId: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(person, FactoryMap(s"Init-$wpId$inShiftId"), startTime) {

  override protected def generateInitialActions(): Unit = {
    waitRandom(0 minutes, 10 minutes)

    move("InFrontOfMainGate")
    waitTillAfterStart(20 minutes)
    move("MainGate")
    accessDoor()
    move("Dispenser")
    waitTillAfterStart(30 minutes)
    retrieveHeadGear()
    move(s"JunctionToWorkPlaceGate-$wpId")
    move(s"InFrontOfWorkPlaceGate-$wpId")
    waitTillAfterStart(40 minutes)
    move(s"WorkPlaceGate-$wpId")
    accessDoor()
    move(s"InWorkPlace-$wpId$inShiftId")

    waitTillAfterStart(9 hours)

    waitRandom(2 minutes, 5 minutes)
    move(s"WorkPlaceGate-$wpId")
    accessDoor()
    move(s"InFrontOfWorkPlaceGate-$wpId")
    move(s"JunctionToWorkPlaceGate-$wpId")

    move("Dispenser")
    returnHeadGear()

    move("MainGate")
    accessDoor()
    move("InFrontOfMainGate")
    move(s"Init-$wpId$inShiftId")
  }
}



object SimulatedLateWorkerInShift {
  def props(person: String, wpId: String, inShiftId: String, startTime: LocalDateTime) = Props(new SimulatedLateWorkerInShift(person, wpId, inShiftId, startTime))
}

// Pos-InWorkPlace1-1, Pos-JunctionToWorkPlace1and2, Pos-InFrontOfMainGate, Pos-Init3-2, Pos-Init2-1, Pos-InWorkPlace2-3, Pos-Gate1,
// Pos-InFrontOfGate2, Pos-InWorkPlace3-3, Pos-BottomRight, Pos-Init1-3, Pos-InWorkPlace3-1, Pos-InFrontOfGate3, Pos-Init3-1, Pos-InWorkPlace2-1, Pos-Gate3,
// Pos-Dispenser, Pos-MainGate, Pos-InWorkPlace1-3, Pos-Init1-2, Pos-Init2-3, Pos-InWorkPlace3-2, Pos-InitStandby-1, Pos-TopLeft, Pos-InWorkPlace1-2, Pos-Init1-1,
// Pos-InWorkPlace2-2, Pos-Init2-2, Pos-Gate2, Pos-InFrontOfGate1, Pos-Init3-3

class SimulatedLateWorkerInShift(person: String, val wpId: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(person, FactoryMap(s"Init-$wpId$inShiftId"), startTime) {

  override protected def generateInitialActions(): Unit = {
    waitRandom(25 minutes, 30 minutes)

    move("InFrontOfMainGate")
    move(s"Init-$wpId$inShiftId")
  }
}


object SimulatedStandbyInShift {
  def props(person: String, inShiftId: String, startTime: LocalDateTime) = Props(new SimulatedStandbyInShift(person, inShiftId, startTime))
}

class SimulatedStandbyInShift(person: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(person, FactoryMap(s"Init-S$inShiftId"), startTime) {

  var forWpId: String = _
  var forInShiftId: String = _

  override protected def generateInitialActions(): Unit = {
    forWpId = null
    forInShiftId = null
  }

  override protected def generateActions(): Unit = {
    if (forWpId == null) {
      // log.info(currentNotifications.toString)
      for (notif <- currentNotifications) {
        notif match {
          case ("workAssigned", List(shiftId, name, wpId, inShiftId)) =>
            this.forWpId = wpId
            this.forInShiftId = inShiftId

            takeOver(name)
        }
      }

      if (forWpId != null) {
        move("InFrontOfMainGate")
        move("MainGate")
        accessDoor()
        move("Dispenser")
        retrieveHeadGear()
        move(s"JunctionToWorkPlaceGate-$forWpId")
        move(s"InFrontOfWorkPlaceGate-$forWpId")
        move(s"WorkPlaceGate-$forWpId")
        accessDoor()
        move(s"InWorkPlace-$forWpId$forInShiftId")

        waitTillAfterStart(9 hours)

        waitRandom(2 minutes, 5 minutes)
        move(s"WorkPlaceGate-$forWpId")
        accessDoor()
        move(s"InFrontOfWorkPlaceGate-$forWpId")
        move(s"JunctionToWorkPlaceGate-$forWpId")
        move("Dispenser")
        returnHeadGear()
        move("MainGate")
        accessDoor()
        move("InFrontOfMainGate")
        move(s"Init-S$inShiftId")
      }
    }
  }
}


object UserControlledWorkerInShift {
  def props(person: String, wpId: String, inShiftId: String, startTime: LocalDateTime) = Props(new UserControlledWorkerInShift(person, wpId, inShiftId, startTime))
}

class UserControlledWorkerInShift(person: String, val wpId: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(person, FactoryMap(s"Init-$wpId$inShiftId"), startTime) {

  object Stage extends Enumeration {
    val MOVING_TO_FACTORY = Value
    val IN_FRONT_OF_FACTORY = Value
    val MOVING_TO_DISPENSER = Value
    val IN_FRONT_OF_DISPENSER = Value
    val MOVING_TO_WORKPLACE = Value
    val IN_FRONT_OF_WORKPLACE = Value
    val FINISHING_SHIFT = Value
    val RETURNING = Value
    type Stage = Value
  }

  import Stage._
  var stage: Stage = MOVING_TO_FACTORY

  override protected def generateInitialActions(): Unit = {
    move("InFrontOfMainGate")
    move("MainGate")

    stage = MOVING_TO_FACTORY
  }

  override protected def generateActions(): Unit = {
    stage match {
      case MOVING_TO_FACTORY =>
        stage = IN_FRONT_OF_FACTORY

      case MOVING_TO_DISPENSER =>
        stage = IN_FRONT_OF_DISPENSER

      case MOVING_TO_WORKPLACE =>
        stage = IN_FRONT_OF_WORKPLACE

      case _ =>
    }
  }

  override protected def processAccess(workerPermissions: List[Permission]): Unit = {
    def validateAccess(verb: String, obj: String): Boolean = {
      if (workerPermissions.exists({
        case AllowPermission(`person`, `verb`, `obj`) => true
        case _ => false
      })) {
        sender() ! AccessResult("allowed")
        true
      } else {
        sender() ! AccessResult("not allowed")
        false
      }
    }

    stage match {
      case IN_FRONT_OF_FACTORY =>
        if (validateAccess("enter", "factory")) {
          accessDoor()
          move("Dispenser")
          stage = MOVING_TO_DISPENSER

        } else {
          move(s"Init-$wpId$inShiftId")
          stage = RETURNING
        }

      case IN_FRONT_OF_DISPENSER =>
        if (validateAccess("use", "dispenser")) {
          retrieveHeadGear()
          move(s"JunctionToWorkPlaceGate-$wpId")
          move(s"InFrontOfWorkPlaceGate-$wpId")
          move(s"WorkPlaceGate-$wpId")
          stage = MOVING_TO_WORKPLACE

        } else {
          move("MainGate")
          move("InFrontOfMainGate")
          move(s"Init-$wpId$inShiftId")
          stage = RETURNING
        }

      case IN_FRONT_OF_WORKPLACE =>
        if (validateAccess("enter", "workplace-A")) {
          accessDoor()
          move(s"InWorkPlace-$wpId$inShiftId")
          waitTillAfterStart(9 hours)
          waitRandom(2 minutes, 5 minutes)
          move(s"WorkPlaceGate-$wpId")
          accessDoor()
          move(s"InFrontOfWorkPlaceGate-$wpId")
          move(s"JunctionToWorkPlaceGate-$wpId")
          move("Dispenser")
          returnHeadGear()
          move("MainGate")
          accessDoor()
          move("InFrontOfMainGate")
          move(s"Init-$wpId$inShiftId")
          stage = FINISHING_SHIFT

        } else {
          move(s"InFrontOfWorkPlaceGate-$wpId")
          move(s"JunctionToWorkPlaceGate-$wpId")
          move("Dispenser")
          returnHeadGear()
          move("MainGate")
          accessDoor()
          move("InFrontOfMainGate")
          move(s"Init-$wpId$inShiftId")
          stage = RETURNING
        }

      case _ =>
        sender() ! AccessResult("none")
    }

  }

}

