package ftnn.k4case

import ftnn.MarshallersSupport
import ftnn.enforcer.tcof._
import spray.json._

import java.time.LocalDateTime
import scala.collection.mutable

trait SerializableEnsemble extends Ensemble {
  def toJson: JsValue
}

case class ScenarioSpec(
                             workersPerWorkplaceCount: Int,
                             workersOnStandbyCount: Int,
                             workersLateCount: Int,
                             startTime: LocalDateTime
                           )

case class Position(x: Double, y: Double)
case class Area(left: Double, top: Double, right: Double, bottom: Double) {
  def this(topLeft: Position, bottomRight: Position) = this(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
  def contains(pos: Position): Boolean = pos != null && pos.x >= left && pos.y >= top && pos.x <= right && pos.y <= bottom
}

case class ScenarioEvent(timestamp: LocalDateTime, eventType: String, person: String, position: Position, params: List[String])

trait WithId {
  def id: String
}


case object Use extends PermissionVerb {
  override def toString = "use"
}

case object Enter extends PermissionVerb {
  override def toString = "enter"
}

case class Read(field: String) extends PermissionVerb {
  override def toString = s"read($field)"
}


class Scenario(scenarioParams: ScenarioSpec) extends Model with ModelGenerator with MarshallersSupport {

  implicit class RichLocalDateTime(val self: LocalDateTime) {
    def isEqualOrBefore(other: LocalDateTime) = !self.isAfter(other)
    def isEqualOrAfter(other: LocalDateTime) = !self.isBefore(other)
  }

  val startTimestamp = scenarioParams.startTime
  var now = startTimestamp

  case class WorkerPotentiallyLateNotification(shift: Shift, worker: Worker) extends Notification("workerPotentiallyLate", List(shift.id, worker.name))

  case class AssignmentCanceledNotification(shift: Shift) extends Notification("assignmentCanceled", List(shift.id))
  case class WorkAssignedNotification(shift: Shift, replacedWorker: Worker) extends Notification("workAssigned", List(shift.id, replacedWorker.name, replacedWorker.wpId, replacedWorker.inShiftId))

  class Door(
              val id: String,
            ) extends Component with WithId {
    name(s"Door ${id}")

    override def toString = s"Door($id)"
  }

  class Dispenser(
                   val id: String,
                 ) extends Component with WithId {
    name(s"Protection equipment dispenser ${id}")

    override def toString = s"Dispenser($id)"
  }

  class Machine(
                   val id: String,
                 ) extends Component with WithId {
    name(s"Machine ${id}")

    override def toString = s"Machine ($id)"
  }

  class Worker(
                val id: String,
                var position: Position,
                var wpId: String,
                var inShiftId: String,
                var hasHeadGear: Boolean
              ) extends Component with WithId {
    name(s"Worker ${id}")

    override def toString = s"Worker($id, $position)"

    def isAt(room: Room) = room.area.contains(position)

    def toJson: JsValue = JsObject(
      "type" -> "Worker".toJson,
      "id" -> id.toJson,
      "position" -> position.asInstanceOf[Position].toJson,
      "wpId" -> wpId.asInstanceOf[String].toJson,
      "inShiftId" -> inShiftId.asInstanceOf[String].toJson,
      "hasHeadGear" -> hasHeadGear.asInstanceOf[Boolean].toJson
    )
  }

  abstract class Room(
              val id: String,
              val area: Area,
              val entryDoor: Door
            ) extends Component with WithId {
    name(s"Room ${id}")
  }

  class WorkPlace(
                   id: String,
                   area: Area,
                   entryDoor: Door,
                   val machine: Machine
                 ) extends Room(id, area, entryDoor) {
    name(s"WorkPlace ${id}")

    var factory: Factory = _

    override def toString = s"WorkPlace($id)"
  }

  class Factory(
                 id: String,
                 area: Area,
                 entryDoor: Door,
                 val dispenser: Dispenser,
                 val workPlaces: List[WorkPlace]
               ) extends Room(id, area, entryDoor) {
    name(s"Factory ${id}")

    for (workPlace <- workPlaces) {
      workPlace.factory = this
    }

    override def toString = s"Factory($id)"
  }

  class Shift(
               val id: String,
               val startTime: LocalDateTime,
               val endTime: LocalDateTime,
               val workPlace: WorkPlace,
               val foreman: Worker,
               val workers: List[Worker],
               val standbys: List[Worker],
               val assignments: Map[Worker, String]
             ) extends Component with WithId {
    name(s"Shift ${id}")

    override def toString = s"Shift($startTime, $endTime, $workPlace, $foreman, $workers, $standbys, $assignments)"
  }


  import ModelDSL._
  val (factory, workersMap, shiftsMap) = withModel { implicit builder =>
    for (wp <- List("A", "B", "C")) {
      val foremanId = s"$wp-foreman"
      withWorker(foremanId, wp, "F")

      var workersInShift = mutable.ListBuffer.empty[String]
      for (idx <- 1 to scenarioParams.workersPerWorkplaceCount) {
        val id = f"$wp%s-worker-$idx%03d"
        withWorker(id, wp, idx.toString)
        workersInShift += id
      }

      var workersOnStandby = mutable.ListBuffer.empty[String]
      for (idx <- 1 to scenarioParams.workersOnStandbyCount) {
        val id = f"$wp%s-standby-$idx%03d"
        withWorker(id, wp, s"S${idx.toString}")
        workersOnStandby += id
      }

      withShift(
        wp,
        startTimestamp plusHours 1,
        startTimestamp plusHours 9,
        wp,
        foremanId,
        workersInShift.toList,
        workersOnStandby.toList,
        workersInShift.map(wrk => (wrk, "A")).toMap
      )
    }
  }


  class FactoryTeam(factory: Factory) extends RootEnsemble {
    name(s"Factory team ${factory.id}")

    class ShiftTeam(shift: Shift) extends SerializableEnsemble {
      name(s"Shift team ${shift.id}")

      // These are like invariants at a given point of time
      val calledInStandbys = shift.standbys.filter(wrk => wrk.notifiedExt { case WorkAssignedNotification(`shift`, _) => true })
      val availableStandbys = shift.standbys diff calledInStandbys

      val canceledWorkers = shift.workers.filter(wrk => wrk notified AssignmentCanceledNotification(shift))
      val canceledWorkersWithoutStandby = canceledWorkers.filterNot(wrk => calledInStandbys.exists(standby => standby.notifiedExt { case WorkAssignedNotification(`shift`, `wrk`) => true }))

      val assignedWorkers = (shift.workers concat calledInStandbys) diff canceledWorkers
      val assignedWorkersWithoutStandbys = shift.workers diff canceledWorkers


      object AccessToFactory extends SerializableEnsemble { // Kdyz se constraints vyhodnoti na LogicalBoolean, tak ten ensemble vubec nezatahujeme solver modelu a poznamename si, jestli vysel nebo ne
        name(s"AccessToFactory")

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 45)) &&
            (now isEqualOrBefore (shift.endTime plusMinutes 45))
        }

        allow(shift.foreman, Enter, shift.workPlace.factory)
        allow(assignedWorkers, Enter, shift.workPlace.factory)

        def toJson: JsValue = JsObject(
          "type" -> "ShiftTeam".toJson
        )
      }

      object AccessToDispenser extends SerializableEnsemble {
        name(s"AccessToDispenser")

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 40)) &&
            (now isEqualOrBefore shift.endTime)
        }

        allow(shift.foreman, Use, shift.workPlace.factory.dispenser)
        allow(assignedWorkers, Use, shift.workPlace.factory.dispenser)

        def toJson: JsValue = JsObject(
          "type" -> "AccessToDispenser".toJson
        )
      }

      object AccessToWorkplace extends SerializableEnsemble {
        name(s"AccessToWorkplace")

        val workersWithHeadGear = (shift.foreman :: assignedWorkers).filter(wrk => wrk.hasHeadGear)

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 25)) &&
            (now isEqualOrBefore (shift.endTime plusMinutes 25))
        }

        allow(workersWithHeadGear, Enter, shift.workPlace)

        def toJson: JsValue = JsObject(
          "type" -> "AccessToWorkplace".toJson
        )
      }

      object AccessToMachine extends SerializableEnsemble {
        name(s"AccessToMachine")

        val workersAtWorkplace = shift.foreman :: assignedWorkers

        situation {
          (now isEqualOrAfter shift.startTime) && (now isEqualOrBefore shift.endTime)
        }

        workersAtWorkplace.foreach(wrk => notify(shift.foreman, WorkerPotentiallyLateNotification(shift, wrk)))

        allow(workersAtWorkplace, Read("aggregatedTemperature"), shift.workPlace.machine)
        allow(workersAtWorkplace, Read("temperature"), shift.workPlace.machine)

        def toJson: JsValue = JsObject(
          "type" -> "AccessToMachine".toJson
        )
      }


      object NotificationAboutWorkersThatArePotentiallyLate extends SerializableEnsemble {
        name(s"NotificationAboutWorkersThatArePotentiallyLate")

        val workersThatAreLate = assignedWorkersWithoutStandbys.filter(wrk => !(wrk isAt shift.workPlace.factory))

        situation {
          now isEqualOrAfter (shift.startTime minusMinutes 25)
        }

        workersThatAreLate.foreach(wrk => notify(shift.foreman, WorkerPotentiallyLateNotification(shift, wrk)))

        allow(shift.foreman, Read("phoneNo"), workersThatAreLate)
        allow(shift.foreman, Read("distanceToWorkPlace"), workersThatAreLate)

        def toJson: JsValue = JsObject(
          "type" -> "NotificationAboutWorkersThatArePotentiallyLate".toJson
        )
      }


      object CancellationOfWorkersThatAreLate extends SerializableEnsemble {
        name(s"CancellationOfWorkersThatAreLate")

        val workersThatAreLate = assignedWorkersWithoutStandbys.filter(wrk => !(wrk isAt shift.workPlace.factory))

        situation {
          now isEqualOrAfter (shift.startTime minusMinutes 15)
        }

        notifyMany(workersThatAreLate, AssignmentCanceledNotification(shift))

        def toJson: JsValue = JsObject(
          "type" -> "CancellationOfWorkersThatAreLate".toJson
        )
      }

      object AssignmentOfStandbys extends SerializableEnsemble {
        name(s"AssignmentOfStandbys")

        class StandbyAssignment(canceledWorker: Worker, standby: Worker) extends SerializableEnsemble {
          name(s"StandbyAssignment for ${canceledWorker.id}")
          notify(standby, WorkAssignedNotification(shift, canceledWorker))

          def toJson: JsValue = JsObject(
            "type" -> "CancellationOfWorkersThatAreLate".toJson,
            "workerId" -> canceledWorker.id.toJson,
            "standById" -> standby.id.toJson
          )
        }

        val standByAssignments = rules(
          (canceledWorkersWithoutStandby zip availableStandbys) map { case (worker,standby) => new StandbyAssignment(worker, standby) }
        )

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 15)) &&
          (now isEqualOrBefore shift.endTime)
        }

        def toJson: JsValue = JsObject(
          "type" -> "AssignmentOfStandbys".toJson,
          "standByAssignments" -> standByAssignments.selectedMembers.map(_.toJson).toJson
        )
      }

      val allEnsembles = rules(
        // Grants
        AccessToFactory,
        AccessToDispenser,
        AccessToWorkplace,
        AccessToMachine,
        NotificationAboutWorkersThatArePotentiallyLate,
        CancellationOfWorkersThatAreLate,
        AssignmentOfStandbys
      )

      def toJson: JsValue = JsObject(
        "type" -> "ShiftTeam".toJson,
        "shiftId" -> shift.id.toJson,
        "allEnsembles" -> allEnsembles.selectedMembers.map(_.toJson).toJson
      )

    }

    val shiftTeams = rules(shiftsMap.values.filter(shift => shift.workPlace.factory == factory).map(shift => new ShiftTeam(shift)))

    def toJson: JsValue = JsObject(
      "type" -> "FactoryTeam".toJson,
      "factoryId" -> factory.id.toJson,
      "shiftTeams" -> shiftTeams.selectedMembers.map(_.toJson).toJson
    )
  }

  val factoryTeam = root(new FactoryTeam(factory))
}

object Scenario {
  def createScenarioSpec(workersPerWorkplaceCount: Int, workersOnStandbyCount: Int, workersLateCount: Int, startTime: LocalDateTime) = {
    ScenarioSpec(
      workersPerWorkplaceCount = workersPerWorkplaceCount,
      workersOnStandbyCount = workersOnStandbyCount,
      workersLateCount = workersLateCount,
      startTime = startTime
    )
  }
}
