package trust40.k4case

import java.time.{Duration, LocalDateTime}

import scala.collection.mutable
import scala.util.Random

trait ModelGenerator {
  this: TestScenario =>

  object ModelDSL {

    private class ModelBuilder {
      val workersMap = mutable.Map.empty[String, Worker]
      val shiftsMap = mutable.Map.empty[String, Shift]

      val workPlacesMap = Map((
        for (wpId <- List("A", "B", "C"))
          yield wpId -> new WorkPlace(
            s"workplace-$wpId",
            new Area(FactoryMap(s"WorkPlace-$wpId-TL"), FactoryMap(s"WorkPlace-$wpId-BR")),
            new Door(s"gate-$wpId"),
            new Machine(s"machine-$wpId")
          )
        ): _*)


      val factory = new Factory(
      "factory",
        new Area(FactoryMap("Factory-TL"), FactoryMap("Factory-BR")),
        new Door("main-gate"),
        new Dispenser("dispenser"),
        workPlacesMap.values.toList
      )

      def addWorker(id: String, wpId: String, inShiftId: String, caps: Set[String]): Unit = {
        workersMap(id) = new Worker(id, null, wpId, inShiftId, caps, false)
      }

      def addShift(id: String, startTime: LocalDateTime, endTime: LocalDateTime, workPlace: String, foreman: String, workers: List[String], standbys: List[String], assignments: Map[String, String]): Unit = {

        shiftsMap(id) = new Shift(
          id,
          startTime,
          endTime,
          workPlacesMap(workPlace),
          workersMap(foreman),
          workers.map(wrk => workersMap(wrk)),
          standbys.map(wrk => workersMap(wrk)),
          assignments.map(keyVal => (workersMap(keyVal._1) -> keyVal._2))
        )
      }
    }

    def withWorker(id: String, wpId: String, inShiftId: String, caps: Set[String])(implicit ms: ModelBuilder): Unit = ms.addWorker(id, wpId, inShiftId, caps)

    def withShift(id: String, startTime: LocalDateTime, endTime: LocalDateTime, workPlace: String, foreman: String, workers: List[String], standbys: List[String], assignments: Map[String, String])(implicit ms: ModelBuilder): Unit =
      ms.addShift(id, startTime, endTime, workPlace, foreman, workers, standbys, assignments)


    def withModel(ops: ModelBuilder => Unit): (Factory, Map[String, Worker], Map[String, Shift]) = {
      val builder = new ModelBuilder
      ops(builder)

      (builder.factory, builder.workersMap.toMap, builder.shiftsMap.toMap)
    }
  }
}

