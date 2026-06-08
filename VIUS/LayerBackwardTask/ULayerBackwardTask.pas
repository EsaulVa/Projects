unit ULayerBackwardTask;

interface

{$REGION ' uses '}
uses // В АЛФАВИТНОМ ПОРЯДКЕ!
{$I '..\..\..\Common Files\GeneralUses.pas'},
  Classes,
  ULayer;
{$ENDREGION}

type

  TLayerBackwardTask = class(TLayer, ILayerBackwardTask, IMenuItem)
    {$REGION ' Class methods '}
    public
      class function ClassTitle: string; override;
    {$ENDREGION ' Class methods '}
    strict private
      FTrajectory: ITrajectory;
      FTangentCalculator: ITangentCalculator;
      FdSdZCalculator: IdSdZCalculator;
      FIntergrator: ISolutionMethodOfSODE;
      FWP0: TWindPoint;
      // Только для отладки расчета витка
//      FK: Extended; // Коэффициент
//      FdV0: Extended;
//      FdU0: Extended;
      FDeltaZ: Extended;
      FPercentage: Extended;
      {$REGION ' Акцессоры '}
      function GetTrajectory: ITrajectory;
      procedure SetTrajectory(const Value: ITrajectory);
      function GetIntegrator: ISolutionMethodOfSODE;
      procedure SetIntegrator(const Value: ISolutionMethodOfSODE);
      function GetWP0: TWindPoint;
      procedure SetWP0(const Value: TWindPoint);
      function GetDeltaZ: Extended;
      procedure SetDeltaZ(const Value: Extended);
      function GetPercentage: Extended;
      procedure SetPercentage(const Value: Extended);
      {$ENDREGION ' Акцессоры '}
      procedure InitIntegrator;
    strict protected
      function ImportInner: Boolean; override;
      function IsBase(const Instance: TObject): Boolean; override;
      function InnerCalculate: Boolean; override;
    public
      constructor Create(AOwner: TComponent); override;
      destructor Destroy; override;
      function CalculateRightParts(const x: TSoEVector; const t: Extended;
        var dx_dt: TSoEVector): Boolean;
    published
      property Trajectory: ITrajectory read GetTrajectory write SetTrajectory;
      property Integrator: ISolutionMethodOfSODE read GetIntegrator write SetIntegrator;
      property WP0: TWindPoint read GetWP0 write SetWP0;
      property DeltaZ: Extended read GetDeltaZ write SetDeltaZ;
      property Percentage: Extended read GetPercentage write SetPercentage;
  end;

implementation

{$REGION ' uses '}
uses // В АЛФАВИТНОМ ПОРЯДКЕ!
  SysUtils,
  UdSdZCalculator,
  ULayerBackwardTaskTF,
  USolutionMethodsOfSoE,
  UTangentCalculator;
{$ENDREGION ' uses '}

{$REGION ' resourcestring '}
resourcestring
  rsLayerBackwardTask = 'Восстанавливаемый';
{$ENDREGION ' resourcestring '}

{ TLayerBackwardTask }

function TLayerBackwardTask.CalculateRightParts(const x: TSoEVector;
  const t: Extended; var dx_dt: TSoEVector): Boolean;
{ Фазовых перменных всего 3 штуки ("u", "v" и "s") }
var
  ds_dz: Extended;
  wp: RWindPoint;
begin
  Result := False;
  try
    wp.u := x[1];
    wp.v := x[2];
    if not FTangentCalculator.Calculate(t, FSurface, FTrajectory.R(t), wp) then Exit;
    with wp do
      ds_dz := FdSdZCalculator.Calculate(t, FDeltaZ, FPercentage, Surface,
        FTrajectory.R(t), FTrajectory.R1(t), wp);
    dx_dt[1] := wp.du * ds_dz;
    dx_dt[2] := wp.dv * ds_dz;
    dx_dt[3] := ds_dz;
  except
  end;
  Result := True;
end;

class function TLayerBackwardTask.ClassTitle: string;
begin
  Result := rsLayerBackwardTask;
end;

constructor TLayerBackwardTask.Create(AOwner: TComponent);
begin
  inherited;
  FTuneFormClass := TLayerBackwardTaskTF;
  FDeltaZ := 150;
  FPercentage := 50;
  FIntergrator := TEulerSM.Create(Self);
  InitIntegrator;
  FTangentCalculator := TTangentCalculatorI.Create(nil);
  FdSdZCalculator := TdSdZCalculatorII.Create(nil);
  FWP0 := TWindPoint.Create(nil);
end;

destructor TLayerBackwardTask.Destroy;
begin
  FWP0.Free;
  AvadaKedavra(@FIntergrator);
  AvadaKedavra(@FTangentCalculator);
  AvadaKedavra(@FdSdZCalculator);
  inherited;
end;

function TLayerBackwardTask.GetDeltaZ: Extended;
begin
  Result := FDeltaZ;
end;

function TLayerBackwardTask.GetIntegrator: ISolutionMethodOfSODE;
begin
  Result := FIntergrator;
end;

function TLayerBackwardTask.GetPercentage: Extended;
begin
  Result := FPercentage;
end;

function TLayerBackwardTask.GetTrajectory: ITrajectory;
begin
  Result := FTrajectory;
end;

function TLayerBackwardTask.GetWP0: TWindPoint;
begin
  Result := FWP0;
end;

function TLayerBackwardTask.ImportInner: Boolean;
var
  srcl: ILayer;
begin
  Result := inherited;
  if not Result then Exit;
  srcl := SelectLayer(ILayer, Self);
  Result := srcl <> nil;
  if not Result then Exit;
  FTrajectory := SelectToG(ITrajectory, Self);
  Result := FTrajectory <> nil;
  if Result then
  begin
    srcl.WindPoints.First.CloneTo(FWP0);
  end;
  Calculate;
end;

procedure TLayerBackwardTask.InitIntegrator;
begin
  if FIntergrator <> nil then
  begin
    FIntergrator.DerivativesFunc := CalculateRightParts;
    FIntergrator.EquationCount := 3;
  end;
end;

function TLayerBackwardTask.InnerCalculate: Boolean;
//**********************************************************
// Восстановление линии укладки по заданной траектории схода
//**********************************************************
var
  z: Extended;
  i: Integer;
  x0,x1, dx_dt: TSoEVector;
  wp: RWindPoint;
begin
  Result := False;
  try
    // 1. Очистка массива точек
    ClearWindPoints;
    // 3. Подготовка метода интегрирования
    FIntergrator.IntegrationStep := FTrajectory.Smax / (FPointCount - 1);
    // 4. Начальная точка витка известна.
    // Формируем начальные условия для интегрирования
    x0 := NullSoEV;
    x1 := NullSoEV;
    dx_dt := NullSoEV;
    wp := FWP0;
//    wp.U  := wp.U + fdU0;
//    wp.V  := wp.V + fdV0;
    FTangentCalculator.Calculate(0, FSurface, FTrajectory.R(0), wp);
    FWindPoints.Add(wp);
    x0[1] := wp.U;
    x0[2] := wp.V;
    x0[3] := wp.S;
    // 5. В цикле восстанавливаем точки линии укладки
    // проверить: мб от 1 до Count - 1 а не так как сейчас!!!!!!!!!!!!!!!!!!!!!!
    for i := 1 to FPointCount - 1 do
    begin
      z := FIntergrator.IntegrationStep * i;
      // Вычисление очередной точки "x1" на основе предыдущей "x0"
      x1 := FIntergrator.Solve(x0, z);
      wp.U := x1[1];
      wp.V := x1[2];
      wp.S := x1[3];
      FTangentCalculator.Calculate(z, FSurface, FTrajectory.R(z), wp);
      FWindPoints.Add(wp);
      x0 := x1;
    end;
    Result := True;
  except
  end;
end;

function TLayerBackwardTask.IsBase(const Instance: TObject): Boolean;
begin
  Result := inherited or (Instance = FTrajectory.AsObject);
end;

procedure TLayerBackwardTask.SetDeltaZ(const Value: Extended);
begin
  FDeltaZ := Value;
end;

procedure TLayerBackwardTask.SetIntegrator(const Value: ISolutionMethodOfSODE);
begin
  ReplaceInterface(Value, @FIntergrator);
  InitIntegrator;
end;

procedure TLayerBackwardTask.SetPercentage(const Value: Extended);
begin
  FPercentage := Value;
end;

procedure TLayerBackwardTask.SetTrajectory(const Value: ITrajectory);
begin
  ReplaceInterface(Value, @FTrajectory);
end;

procedure TLayerBackwardTask.SetWP0(const Value: TWindPoint);
begin
  Value.CloneTo(FWP0);
end;

initialization
  RegisterClass(TLayerBackwardTask);

end.
