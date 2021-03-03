package trust40

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json.{DefaultJsonProtocol, JsNumber, JsValue, JsonFormat, deserializationError}
import trust40.k4case.{AccessResult, Position, Simulation, SimulationState, ValidateResult, WorkerState}

trait MarshallersSupport extends SprayJsonSupport with DefaultJsonProtocol {
  implicit object SimulationStateStateJsonFormat extends JsonFormat[Simulation.State.State] {
    def write(x: Simulation.State.State) = JsNumber(x.id)
    def read(value: JsValue) = value match {
      case JsNumber(x) => Simulation.State(x.toInt)
      case x => deserializationError("Expected Int as JsNumber, but got " + x)
    }
  }

  implicit val positionFormat = jsonFormat2(Position)
  implicit val workerStateFormat = jsonFormat3(WorkerState)
  implicit val simulationStateFormat = jsonFormat5(SimulationState)
  implicit val accessResultFormat = jsonFormat1(AccessResult)
  implicit val validateResultFormat = jsonFormat1(ValidateResult)
}
