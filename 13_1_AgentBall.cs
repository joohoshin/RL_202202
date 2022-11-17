using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// ml agent 추가
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

// mlagent 참고  https://docs.unity3d.com/Packages/com.unity.ml-agents@2.3/api/index.html

public class AgentBall : Agent  //ml agent의 agent class 상속
{

    Rigidbody rBody; // ball agent
    public Transform Target;  // 목적지인 cube를 입력 받을 변수, public을 사용하여 unity에서 object를 지정할 수 있도록 함

    void Start()
    {
        rBody = GetComponent<Rigidbody>();   // 현재 오브젝트를 변수에 저장
    }

    // Agent 클래스 안의 이 함수를 재정의 함. 에피소드 시작 시 해야할 것들을 넣어 줌 
    // override는 상속받은 클래스의 함수를 다시 작성하는 것 (Subclassing)
    public override void OnEpisodeBegin()  
    {
       // Plane 아래로 공이 갔는 지 확인
        if (this.transform.localPosition.y < 0)  // Transform을 통해서 현재 object의 위치를 가져올 수 있음. this는 현재의 클래스를 의미함
        {
            this.rBody.angularVelocity = Vector3.zero; // 각속도 0으로 초기화
            this.rBody.velocity = Vector3.zero;  // 속도 초기화
            this.transform.localPosition = new Vector3( 0, 0.5f, 0);  // 처음 위치로 초기화
        }

        // 타겟의 위치를 랜덤하게 변경함 (학습때마다 바뀌도록)
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           0.5f,
                                           Random.value * 8 - 4);
    }

    // Observation 값을 지정하는 함수를 작성
    public override void CollectObservations(VectorSensor sensor)  // VectorSensor는 데이터 타입, sensor는 변수명임
    // https://docs.unity3d.com/Packages/com.unity.ml-agents@2.3/api/Unity.MLAgents.Sensors.VectorSensor.html
    {
        // 타겟과 agent의 위치를 obs에 추가
        sensor.AddObservation(Target.localPosition);  // 관측하고 싶은 값을 AddObservation을 통해 추가함
        sensor.AddObservation(this.transform.localPosition);

        // agent의 속도도 추가
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    // 액션을 하고 리워드 계산하는 부분 작성
    public float forceMultiplier = 10;  // 힘을 조정하는 변수, public을 넣으면 unity에서 조정 가능함
    public override void OnActionReceived(ActionBuffers actionBuffers)   // public ActionBuffers(float[] continuousActions, int[] discreteActions)
    {
        // Actions, size = 2
        // Actionbuffers는 action을 저장하는 배열
        // public ActionBuffers(float[] continuousActions, int[] discreteActions)
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        // 타겟까지의 거리를 계산
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // 일정 범위 이내이면, 보상을 주고 종료
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // 판 아래로 떨어지면 실패, 에피소드 종료
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }

    // 직접 키보드 등으로 입력해볼 수 있는 함수
    // Behavior Type 이 Heuristic Only 일때 동작
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");  // 키보드 입력을 받는다
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
    // Update는 사용 안함
    // void Update()
    // {
        
    // }
}


// Decision Requester의 파라미터
// Request Period: 몇 프레임마다 Decision을 요청할지 
// The frequency with which the agent requests a decision. A DecisionPeriod of 5 means that the Agent will request a decision every 5 Academy steps. 
// https://docs.unity3d.com/Packages/com.unity.ml-agents@2.3/api/Unity.MLAgents.DecisionRequester.html

// Behavior Parameters
// https://docs.unity3d.com/Packages/com.unity.ml-agents@2.3/api/Unity.MLAgents.Policies.BehaviorParameters.html
// Behavior Name
// Space Size
// Action Size
// Behavior Type