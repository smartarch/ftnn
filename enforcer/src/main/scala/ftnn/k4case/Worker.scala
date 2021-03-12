package ftnn.k4case

import java.time.temporal.ChronoUnit
import java.time.{Duration, LocalDateTime}
import scala.collection.mutable
import scala.util.Random

abstract class AbstractSimulatedWorker(val randSeed: Int, val name: String, val startTime: LocalDateTime) {
  private var random: Random = _

  var initialPosition: Position = _
  var startPosition: Position = _

  private def initRandom(): Unit = {
    random = new Random(name.hashCode + randSeed)

    val workerStartTopLeft = FactoryMap(s"WorkerStart-TL")
    val workerStartBottomRight = FactoryMap(s"WorkerStart-BR")

    initialPosition = Position(
      workerStartTopLeft.x + random.nextDouble() * (workerStartBottomRight.x - workerStartTopLeft.x),
      workerStartTopLeft.y + random.nextDouble() * (workerStartBottomRight.y - workerStartTopLeft.y)
    )

    startPosition = initialPosition
  }

  initRandom()

  private abstract class Action(val startTime: LocalDateTime, val duration: Duration, val startPosition: Position, val targetPosition: Position) {
    def getEvents(currentTime: LocalDateTime): List[ScenarioEvent] = List()
    def isDue(currentTime: LocalDateTime) = !currentTime.isBefore(startTime)
    def isOver(currentTime: LocalDateTime) = !currentTime.isBefore(startTime.plus(duration))
  }

  private class InitAction() extends Action(startTime, Duration.ZERO, startPosition, startPosition) {
    override def toString: String = s"InitAction($startTime, $startPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(startTime, "init", name, startPosition, List()))
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

      List(ScenarioEvent(startTime.plusNanos(nsSinceStart), "move", name, Position(x, y), List()))
    }
  }

  private class AccessDoorAction(timestamp: LocalDateTime, doorPosition: Position) extends Action(timestamp, Duration.ZERO, doorPosition, doorPosition) {
    override def toString: String = s"AccessDoorAction($timestamp, $doorPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "access-door", name, doorPosition, List()))
  }

  private class TakeOverAction(timestamp: LocalDateTime, position: Position, replacedWorkerId: String) extends Action(timestamp, Duration.ZERO, position, position) {
    override def toString: String = s"TakeOverAction($timestamp, $replacedWorkerId)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "take-over", name, position, List(replacedWorkerId)))
  }

  private class RetrieveHeadGearAction(timestamp: LocalDateTime, dispenserPosition: Position) extends Action(timestamp, Duration.ZERO, dispenserPosition, dispenserPosition) {
    override def toString: String = s"RetrieveHeadGearAction($timestamp, $dispenserPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "retrieve-head-gear", name, dispenserPosition, List()))
  }

  private class ReturnHeadGearAction(timestamp: LocalDateTime, dispenserPosition: Position) extends Action(timestamp, Duration.ZERO, dispenserPosition, dispenserPosition) {
    override def toString: String = s"ReturnHeadGearAction($timestamp, $dispenserPosition)"
    override def getEvents(currentTimestamp: LocalDateTime): List[ScenarioEvent] = List(ScenarioEvent(timestamp, "return-head-gear", name, dispenserPosition, List()))
  }

  private class WaitAction(timestamp: LocalDateTime, duration: Duration, position: Position) extends Action(timestamp, duration, position, position) {
    override def toString: String = s"WaitAction($timestamp, $duration, $position)"
  }

  private val futureActions = mutable.ListBuffer.empty[Action]
  private var currentAction: Action = _

  protected var currentTime: LocalDateTime = _
  protected var currentPosition: Position = _

  protected var currentNotifications: List[(String, List[String])] = List()

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

  protected def moveToPos(targetPosition: Position, maxPace: Double = 9 /* seconds / position unit */, minPace: Double = 11): Unit = {
    val startPosition = lastActionPosition
    val xDistance = startPosition.x - targetPosition.x
    val yDistance = startPosition.y - targetPosition.y
    val distance = Math.sqrt(xDistance * xDistance + yDistance * yDistance)
    val duration = Duration.ofMillis((distance * (maxPace + random.nextDouble() * (minPace - maxPace)) * 1000).toInt)

    futureActions += new MoveAction(lastActionTime, duration, startPosition, targetPosition)
  }

  protected def move(id: String, maxPace: Double = 9 /* seconds / position unit */, minPace: Double = 11): Unit = moveToPos(FactoryMap(id), maxPace, minPace)

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

  def step(currentTime: LocalDateTime, notifications: List[(String, List[String])]): List[ScenarioEvent] = {
    currentNotifications = notifications
    generateEvents(currentTime)
  }

  def reset(): Unit = {
    initRandom()
    futureActions.clear()
    currentAction = null

    currentTime = startTime
    currentPosition = startPosition

    futureActions += new InitAction()
    generateInitialActions()
  }
}


class SimulatedWorkerInShift(randSeed: Int, name: String, val wpId: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(randSeed, name, startTime) {

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
    moveToPos(initialPosition)
  }
}


class SimulatedLateWorkerInShift(randSeed: Int, name: String, val wpId: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(randSeed: Int, name, startTime) {

  override protected def generateInitialActions(): Unit = {
    waitRandom(25 minutes, 30 minutes)

    move("InFrontOfMainGate")
    moveToPos(initialPosition)
  }
}


class SimulatedStandbyInShift(randSeed: Int, name: String, val inShiftId: String, startTime: LocalDateTime)
  extends AbstractSimulatedWorker(randSeed: Int, name, startTime) {

  var forWpId: String = _
  var forInShiftId: String = _

  override protected def generateInitialActions(): Unit = {
    forWpId = null
    forInShiftId = null
  }

  override protected def generateActions(): Unit = {
    if (forWpId == null) {
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
        moveToPos(initialPosition)
      }
    }
  }
}


