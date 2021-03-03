package trust40.k4case

import java.time.LocalDateTime

import trust40.enforcer.tcof._

import scala.collection.mutable

case class TestScenarioSpec(
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


class TestScenario(scenarioParams: TestScenarioSpec) extends Model with ModelGenerator {

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
                val capabilities: Set[String],
                var hasHeadGear: Boolean
              ) extends Component with WithId {
    name(s"Worker ${id}")

    override def toString = s"Worker($id, $position)"

    def isAt(room: Room) = room.area.contains(position)
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
    var workersOnStandby = mutable.ListBuffer.empty[String]
    for (idx <- 1 to scenarioParams.workersOnStandbyCount) {
      val id = f"standby-$idx%03d"
      withWorker(id, null, idx.toString, Set("A", "B", "C", "D", "E"))
      workersOnStandby += id
    }

    for (wp <- List("A", "B", "C")) {
      val foremanId = s"$wp-foreman"
      withWorker(foremanId, wp, "F", Set("A", "B", "C", "D", "E"))

      var workersInShift = mutable.ListBuffer.empty[String]
      for (idx <- 1 to scenarioParams.workersPerWorkplaceCount) {
        val id = f"$wp%s-worker-$idx%03d"
        withWorker(id, wp, idx.toString, Set("A", "B", "C", "D", "E"))
        workersInShift += id
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

    class ShiftTeam(shift: Shift) extends Ensemble {
      name(s"Shift team ${shift.id}")

      // These are like invariants at a given point of time
      val calledInStandbys = shift.standbys.filter(wrk => wrk.notifiedExt { case WorkAssignedNotification(`shift`, _) => true })
      val availableStandbys = shift.standbys diff calledInStandbys

      val canceledWorkers = shift.workers.filter(wrk => wrk notified AssignmentCanceledNotification(shift))
      val canceledWorkersWithoutStandby = canceledWorkers.filterNot(wrk => calledInStandbys.exists(standby => standby.notifiedExt { case WorkAssignedNotification(`shift`, `wrk`) => true }))

      val assignedWorkers = (shift.workers union calledInStandbys) diff canceledWorkers
      val assignedWorkersWithoutStandbys = shift.workers diff canceledWorkers


      object AccessToFactory extends Ensemble { // Kdyz se constraints vyhodnoti na LogicalBoolean, tak ten ensemble vubec nezatahujeme solver modelu a poznamename si, jestli vysel nebo ne
        name(s"AccessToFactory")

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 45)) &&
            (now isEqualOrBefore (shift.endTime plusMinutes 45))
        }

        allow(shift.foreman, Enter, shift.workPlace.factory)
        allow(assignedWorkers, Enter, shift.workPlace.factory)
      }

      object AccessToDispenser extends Ensemble {
        name(s"AccessToDispenser")

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 40)) &&
            (now isEqualOrBefore shift.endTime)
        }

        allow(shift.foreman, Use, shift.workPlace.factory.dispenser)
        allow(assignedWorkers, Use, shift.workPlace.factory.dispenser)
      }

      object AccessToWorkplace extends Ensemble {
        name(s"AccessToWorkplace")

        val workersWithHeadGear = (shift.foreman :: assignedWorkers).filter(wrk => wrk.hasHeadGear)

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 25)) &&
            (now isEqualOrBefore (shift.endTime plusMinutes 25))
        }

        allow(workersWithHeadGear, Enter, shift.workPlace)
      }

      object AccessToMachine extends Ensemble {
        name(s"AccessToMachine")

        val workersAtWorkplace = shift.foreman :: assignedWorkers

        situation {
          (now isEqualOrAfter shift.startTime) && (now isEqualOrBefore shift.endTime)
        }

        workersAtWorkplace.foreach(wrk => notify(shift.foreman, WorkerPotentiallyLateNotification(shift, wrk)))

        allow(workersAtWorkplace, Read("aggregatedTemperature"), shift.workPlace.machine)
        allow(workersAtWorkplace, Read("temperature"), shift.workPlace.machine)
      }


      object NoAccessToMachineSensitiveDataOtherThanFromWorkplace extends Ensemble {
        name(s"NoAccessToMachineSensitiveDataOtherThanFromWorkplace")

        val workersAtWorkplace = (shift.foreman :: assignedWorkers).filter(wrk => !(wrk isAt shift.workPlace))

        deny(workersAtWorkplace, Read("*"), shift.workPlace.machine, PrivacyLevel.SENSITIVE)
      }

      object NotificationAboutWorkersThatArePotentiallyLate extends Ensemble {
        name(s"NotificationAboutWorkersThatArePotentiallyLate")

        val workersThatAreLate = assignedWorkersWithoutStandbys.filter(wrk => !(wrk isAt shift.workPlace.factory))

        situation {
          now isEqualOrAfter (shift.startTime minusMinutes 25)
        }

        workersThatAreLate.foreach(wrk => notify(shift.foreman, WorkerPotentiallyLateNotification(shift, wrk)))

        allow(shift.foreman, Read("phoneNo"), workersThatAreLate)
        allow(shift.foreman, Read("distanceToWorkPlace"), workersThatAreLate)
      }


      object CancellationOfWorkersThatAreLate extends Ensemble {
        name(s"CancellationOfWorkersThatAreLate")

        val workersThatAreLate = assignedWorkersWithoutStandbys.filter(wrk => !(wrk isAt shift.workPlace.factory))

        situation {
          now isEqualOrAfter (shift.startTime minusMinutes 15)
        }

        notifyMany(workersThatAreLate, AssignmentCanceledNotification(shift))
      }

      object AssignmentOfStandbys extends Ensemble {
        name(s"AssignmentOfStandbys")

        class StandbyAssignment(canceledWorker: Worker) extends Ensemble {
          name(s"StandbyAssignment for ${canceledWorker.id}")

          val standby = oneOf(availableStandbys)

          constraints {
            standby.all(_.capabilities contains shift.assignments(canceledWorker))
          }

          notify(standby.selectedMembers.head, WorkAssignedNotification(shift, canceledWorker))
        }

        val standbyAssignments = rules(canceledWorkersWithoutStandby.map(new StandbyAssignment(_)))
        val selectedStandbys = unionOf(standbyAssignments.map(_.standby))

        situation {
          (now isEqualOrAfter (shift.startTime minusMinutes 15)) &&
          (now isEqualOrBefore shift.endTime)
        }

        constraints {
          standbyAssignments.map(_.standby).allDisjoint
        }
      }

      object NoAccessToPersonalDataExceptForLateWorkers extends Ensemble {
          name(s"NoAccessToPersonalDataExceptForLateWorkers")

          val workersPotentiallyLate =
              if ((now isEqualOrAfter (shift.startTime minusMinutes 25)) && (now isEqualOrBefore shift.startTime))
                  assignedWorkers.filter(wrk => !(wrk isAt shift.workPlace.factory))
              else
                  Nil

          val workers = shift.workers diff workersPotentiallyLate

          deny(shift.foreman, Read("*"), workers, PrivacyLevel.PUBLIC)
          deny(shift.foreman, Read("*"), workersPotentiallyLate, PrivacyLevel.SENSITIVE)
      }

      rules(
        // Grants
        AccessToFactory,
        AccessToDispenser,
        AccessToWorkplace,
        AccessToMachine,
        NotificationAboutWorkersThatArePotentiallyLate,
        CancellationOfWorkersThatAreLate,
        AssignmentOfStandbys,

        // Assertions
        NoAccessToPersonalDataExceptForLateWorkers,
        NoAccessToMachineSensitiveDataOtherThanFromWorkplace
      )
    }

    val shiftTeams = rules(shiftsMap.values.filter(shift => shift.workPlace.factory == factory).map(shift => new ShiftTeam(shift)))

    constraints {
      shiftTeams.map(shift => shift.AssignmentOfStandbys.selectedStandbys).allDisjoint
    }
  }

  val factoryTeam = root(new FactoryTeam(factory))
}

object TestScenario {
  def createScenarioSpec(workersPerWorkplaceCount: Int, workersOnStandbyCount: Int, workersLateCount: Int, startTime: LocalDateTime) = {
    TestScenarioSpec(
      workersPerWorkplaceCount = workersPerWorkplaceCount,
      workersOnStandbyCount = workersOnStandbyCount,
      workersLateCount = workersLateCount,
      startTime = startTime
    )
  }
}
